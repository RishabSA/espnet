import torch
import whisper
from espnet2.asr.decoder.whisper_decoder import OpenAIWhisperDecoder
from espnet2.asr.encoder.whisper_encoder import OpenAIWhisperEncoder
from espnet2.legacy.nets.batch_beam_search import BatchBeamSearch
from whisper.tokenizer import Tokenizer, get_tokenizer

sample_rate = 16000
# sequences per teacher-forced rescoring forward; vocab-wide logits are ~40 MB per
# sequence of length 200, so this bounds peak memory regardless of batch x n-best
rescore_batch = 8


def pick_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def tokenizer_for_vocab(n_vocab: int, language: str = "en") -> Tokenizer:
    # whisper's own vocab arithmetic: multilingual vocabs are >= 51865 and the
    # language-token count is what remains after the 51765 text/eot tokens
    multilingual = n_vocab >= 51865
    num_languages = n_vocab - 51765 - int(multilingual)
    return get_tokenizer(
        multilingual=multilingual,
        num_languages=num_languages,
        language=language if multilingual else None,
        task="transcribe" if multilingual else None,
    )


class _SuppressingWhisperDecoder(OpenAIWhisperDecoder):
    # the espnet wrapper scores the raw vocab; whisper decoding masks a standing
    # suppression set plus blank/eot at the first content position, and beam search
    # calls into batch_score, so the mask lives here (engine fills the attributes)
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.suppress_ids = []
        self.blank_ids = []
        self.sample_begin = 0

    def batch_score(
        self, ys: torch.Tensor, states: list, xs: torch.Tensor
    ) -> tuple[torch.Tensor, list]:
        logp, states = super().batch_score(ys, states, xs)
        logp[:, self.suppress_ids] = float("-inf")
        if ys.size(dim=1) == self.sample_begin:
            logp[:, self.blank_ids] = float("-inf")
        return logp, states


class _WhisperBeamSearch(BatchBeamSearch):
    # espnet's fixed-maxlen search runs every chunk to the length cap, so junk
    # continuations of already-decoded speech keep growing and force-ended repetition
    # loops reach the final ranking, where a confident loop can out-score the correct
    # short hypothesis under length normalization; whisper stops once beam_size
    # hypotheses have finished (patience 1.0), so replicate that by emptying the
    # running set, which ends espnet's search loop
    def post_process(self, i, maxlen, minlen, maxlenratio, running_hyps, ended_hyps):
        remained = super().post_process(i, maxlen, minlen, maxlenratio, running_hyps, ended_hyps)
        if len(ended_hyps) >= self.beam_size:
            return self.batchfy([])
        return remained


class WhisperEngine:
    def __init__(
        self,
        model_id: str,
        device: torch.device | None = None,
        download_dir: str | None = None,
    ) -> None:
        self.device = device if device is not None else pick_device()

        # one throwaway cpu load to learn the checkpoint dims: the espnet wrappers
        # each reload the checkpoint internally, and the decoder wrapper silently
        # reinitializes the token embedding if vocab_size is not exact
        probe = whisper.load_model(model_id, device="cpu", download_root=download_dir)
        self.dims = probe.dims
        del probe

        self.encoder = OpenAIWhisperEncoder(
            whisper_model=model_id, download_dir=download_dir, do_pad_trim=True
        ).to(self.device)
        self.decoder = _SuppressingWhisperDecoder(
            vocab_size=self.dims.n_vocab,
            encoder_output_size=self.dims.n_audio_state,
            whisper_model=model_id,
            download_dir=download_dir,
        ).to(self.device)
        # the wrappers call .train() during construction
        self.encoder.eval()
        self.decoder.eval()

        self.tokenizer = tokenizer_for_vocab(self.dims.n_vocab)
        # whisper's default suppression set (DecodingTask with suppress_tokens="-1"):
        # non-speech symbols plus the sot/task specials; eot stays allowed, and
        # timestamps are already discouraged by <|notimestamps|> in the primer
        suppress = set(self.tokenizer.non_speech_tokens)
        specials = self.tokenizer.special_tokens
        for name in (
            "<|startoftranscript|>", "<|startofprev|>", "<|startoflm|>",
            "<|transcribe|>", "<|translate|>", "<|nospeech|>",
        ):
            if name in specials:
                suppress.add(specials[name])
        self.decoder.suppress_ids = sorted(suppress)
        self.decoder.blank_ids = [self.tokenizer.encode(" ")[0], self.tokenizer.eot]

        self.model_id = model_id

    def _autocast(self) -> torch.amp.autocast:
        # weights stay fp32; matmuls run fp16 on cuda, everything stays fp32 elsewhere
        return torch.amp.autocast(
            self.device.type, dtype=torch.float16, enabled=self.device.type == "cuda"
        )

    def _tokenizer_for(self, language: str) -> Tokenizer:
        if language == "en":
            return self.tokenizer
        return tokenizer_for_vocab(self.dims.n_vocab, language)

    def _primer(self, language: str, prompt: str | None) -> list[int]:
        tokenizer = self._tokenizer_for(language)
        primer = list(tokenizer.sot_sequence_including_notimestamps)
        if prompt is not None:
            # whisper convention: prompt sits behind <|startofprev|> and occupies at
            # most half the text context
            prompt_ids = tokenizer.encode(" " + prompt.strip())
            keep = self.dims.n_text_ctx // 2 - 1
            primer = [tokenizer.special_tokens["<|startofprev|>"]] + prompt_ids[-keep:] + primer
        return primer

    def _encode(self, audios: list[torch.Tensor]) -> torch.Tensor:
        lens = torch.tensor([a.shape[0] for a in audios], device=self.device)
        batch = torch.zeros(len(audios), max(a.shape[0] for a in audios))
        for i, a in enumerate(audios):
            batch[i, : a.shape[0]] = a

        with torch.inference_mode(), self._autocast():
            enc, _, _ = self.encoder(batch.to(self.device), lens)
        # fp32 memory keeps beam-search score math exact under autocast
        return enc.float()  # shape: (batch, 1500, d_model)

    def _token_logprobs(self, memory: torch.Tensor, sequences: torch.Tensor) -> torch.Tensor:
        # per-token logprobs from a batched teacher-forced forward: puts pass-1
        # confidences on exactly the same raw-model scale score_text uses, which is
        # what delta_null comparisons assume; the forward sees the full sequence
        # including any prompt prefix, so logprobs are conditioned as the decode was
        hlens = torch.full((memory.shape[0],), memory.shape[1], device=self.device)
        slens = torch.full((sequences.shape[0],), sequences.shape[1], device=self.device)

        pieces = []
        with torch.inference_mode(), self._autocast():
            for lo in range(0, sequences.shape[0], rescore_batch):
                hi = min(lo + rescore_batch, sequences.shape[0])
                logits, _ = self.decoder(
                    memory[lo:hi], hlens[lo:hi], sequences[lo:hi], slens[lo:hi]
                )
                logprob_dist = torch.log_softmax(logits.float(), dim=-1)
                target = sequences[lo:hi, 1:]  # shape: (b, seq_len - 1)
                pieces.append(
                    logprob_dist[:, :-1].gather(dim=-1, index=target.unsqueeze(dim=-1)).squeeze(dim=-1)
                )
        return torch.cat(pieces, dim=0)  # shape: (n_sequences, seq_len - 1)

    def n_best_decode_batch(
        self,
        audios: list[torch.Tensor],
        num_beams: int = 8,
        num_return_sequences: int = 8,
        language: str = "en",
        prompt: str | None = None,
    ) -> list[list[dict]]:
        tokenizer = self._tokenizer_for(language)
        primer = self._primer(language, prompt)
        self.decoder.sample_begin = len(primer)

        beam = _WhisperBeamSearch(
            scorers={"decoder": self.decoder},
            weights={"decoder": 1.0},
            beam_size=num_beams,
            vocab_size=self.dims.n_vocab,
            sos=tokenizer.sot,
            eos=tokenizer.eot,
            hyp_primer=primer,
            # rank ended hypotheses by length-normalized score, the same objective as
            # whisper/HF beam ranking at length penalty 1.0
            normalize_length=True,
        )
        # the decoder positional table has n_text_ctx rows, and the forced eos of the
        # final search step lands on top of the last searched token, so the longest
        # possible yseq is len(primer) + maxlen + 1; keep that at n_text_ctx exactly
        # or the teacher-forced rescoring forward overflows the table
        maxlen = self.dims.n_text_ctx - len(primer) - 1

        enc = self._encode(audios)
        per_chunk = []
        with torch.inference_mode(), self._autocast():
            for b in range(enc.shape[0]):
                nbest = beam.forward(x=enc[b], maxlenratio=float(-maxlen))
                per_chunk.append(nbest[:num_return_sequences])

        # one flat teacher-forced pass over every retained hypothesis
        yseqs = [hyp.yseq.tolist() for hyps in per_chunk for hyp in hyps]
        chunk_index = [b for b, hyps in enumerate(per_chunk) for _ in hyps]
        max_len = max(len(y) for y in yseqs)
        sequences = torch.full((len(yseqs), max_len), tokenizer.eot, device=self.device)
        for i, y in enumerate(yseqs):
            sequences[i, : len(y)] = torch.tensor(y, device=self.device)
        token_lp = self._token_logprobs(enc[chunk_index], sequences)

        eot = tokenizer.eot
        results = []
        row = 0
        for hyps in per_chunk:
            out = []
            for rank, hyp in enumerate(hyps):
                yseq = yseqs[row]
                # token at position j is predicted by logits at j - 1; keep text
                # tokens only, dropping the primer and every special past end-of-text
                tokens = [
                    {
                        "id": tok_id,
                        "tok": tokenizer.encoding.decode_single_token_bytes(tok_id).decode(
                            "utf-8", errors="replace"
                        ),
                        "logprob": token_lp[row, j - 1].item(),
                    }
                    for j, tok_id in enumerate(yseq)
                    if j >= len(primer) and tok_id < eot
                ]
                sum_logprob = sum(t["logprob"] for t in tokens)
                avg_logprob = sum_logprob / len(tokens) if tokens else 0.0
                out.append(
                    {
                        "rank": rank,
                        "text": tokenizer.decode([t["id"] for t in tokens]).strip(),
                        "tokens": tokens,
                        "sum_logprob": sum_logprob,
                        "avg_logprob": avg_logprob,
                        "beam_score": float(hyp.score) / (len(yseq) - 1),
                    }
                )
                row += 1
            results.append(out)
        return results

    def n_best_decode(
        self,
        audio: torch.Tensor,
        num_beams: int = 8,
        num_return_sequences: int = 8,
        language: str = "en",
        prompt: str | None = None,
    ) -> list[dict]:
        return self.n_best_decode_batch(
            [audio],
            num_beams=num_beams,
            num_return_sequences=num_return_sequences,
            language=language,
            prompt=prompt,
        )[0]

    def score_texts(
        self,
        audio: torch.Tensor,
        texts: list[str],
        focuses: list[tuple[int, int] | None] | None = None,
        language: str = "en",
    ) -> list[dict]:
        # teacher-forced scoring of several texts over one audio window: one encoder pass, the
        # decoder sub-batched; each focus is a character range in its text whose token logprobs
        # are summed separately (spec section 6.8, canonicalizer b)
        tokenizer = self._tokenizer_for(language)
        special = self._primer(language, None)
        # leading space matches Whisper's own output convention for mid-transcript words
        ids = [tokenizer.encode(" " + t.strip()) for t in texts]
        seqs = [special + t + [tokenizer.eot] for t in ids]
        width = max(len(q) for q in seqs)
        # right padding with eot is inert: attention is causal, so real positions never see it
        padded = torch.tensor([q + [tokenizer.eot] * (width - len(q)) for q in seqs], device=self.device)

        enc = self._encode([audio])
        token_lp = self._token_logprobs(enc.expand(len(seqs), -1, -1), padded)  # shape: (n, width - 1)

        results = []
        for k, text_ids in enumerate(ids):
            # text token j occupies decoder position len(special) + j, so target index
            # len(special) + j - 1
            text_lp = token_lp[k, len(special) - 1 : len(special) - 1 + len(text_ids)]
            result = {
                "sum_all": text_lp.sum().item(),
                "mean_all": text_lp.mean().item(),
                "n_tokens": len(text_ids),
            }
            focus = focuses[k] if focuses is not None else None
            if focus is not None:
                # character offsets in the prefixed string, rebuilt from token bytes since
                # tiktoken has no offset mapping; a piece boundary inside a multi-byte
                # character floors to the previous character, which only fuzzes non-ASCII
                starts, ends = [], []
                grown = b""
                for tok_id in text_ids:
                    starts.append(len(grown.decode("utf-8", errors="ignore")))
                    grown += tokenizer.encoding.decode_single_token_bytes(tok_id)
                    ends.append(len(grown.decode("utf-8", errors="ignore")))
                # offsets are relative to the prefixed string, shift the caller's range by 1
                lo, hi = focus[0] + 1, focus[1] + 1
                idx = [j for j, (st, en) in enumerate(zip(starts, ends, strict=True)) if st < hi and en > lo]
                if not idx:
                    raise ValueError(
                        f"focus range {focus} selects no tokens in text of length {len(texts[k])}"
                    )
                focus_lp = text_lp[idx]
                result.update(
                    {
                        "sum_focus": focus_lp.sum().item(),
                        "mean_focus": focus_lp.mean().item(),
                        "n_focus_tokens": len(idx),
                    }
                )
            results.append(result)
        return results

    def score_text(
        self,
        audio: torch.Tensor,
        text: str,
        focus: tuple[int, int] | None = None,
        language: str = "en",
    ) -> dict:
        return self.score_texts(audio, [text], [focus], language)[0]

    def biased_decode(
        self,
        audio: torch.Tensor,
        prompt: str,
        num_beams: int = 8,
        num_return_sequences: int = 8,
        language: str = "en",
    ) -> list[dict]:
        return self.n_best_decode(
            audio,
            num_beams=num_beams,
            num_return_sequences=num_return_sequences,
            language=language,
            prompt=prompt,
        )
