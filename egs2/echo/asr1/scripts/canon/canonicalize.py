import argparse
import json
import math
import os
import re
import statistics
import time
from collections import Counter, defaultdict
from collections.abc import Callable
from datetime import UTC, datetime
from functools import partial
from pathlib import Path

from tqdm import tqdm

from scripts.analysis.e1_windows import stitched_words
from scripts.common.align import align_doc, realize
from scripts.common.audio import load_audio, slice_audio
from scripts.common.io import append_config, read_jsonl
from scripts.mine.mine_candidates import clean_word, possessive_re, word_norm

# canonicalizer (a): posterior weights clipped so a near-zero-confidence mention still counts
weight_floor, weight_ceiling = 0.05, 1.0
edge_re = re.compile(r"^(\W*)(.*?)(\W*)$", re.DOTALL)


def norm_of(surface: str) -> str:
    return " ".join(n for n in (word_norm(w) for w in surface.split()) if n)


def vote(mentions: list[dict]) -> tuple[str, dict[str, float]]:
    # canonicalizer (a): confidence-weighted vote over normalized spellings, alphabetical tie-break
    totals = defaultdict(float)
    for m in mentions:
        totals[m["norm"]] += min(weight_ceiling, max(weight_floor, math.exp(m["conf"])))
    return max(sorted(totals), key=totals.get), dict(totals)


def surfaces_by_norm(mentions: list[dict]) -> dict[str, str]:
    # each normalized spelling's most frequent surface with the possessive removed: the cased
    # form spliced into text; each occurrence re-attaches its own possessive at splice time
    counts = defaultdict(Counter)
    for m in mentions:
        counts[m["norm"]][possessive_re.sub("", m["surface"])] += 1
    return {n: c.most_common(1)[0][0] for n, c in counts.items()}


def possessive_of(mention: dict, words: list[dict]) -> str:
    # the occurrence's own possessive suffix, if any, so every variant is scored with the same
    # morphology the audio carries at that occurrence
    found = possessive_re.search(clean_word(words[mention["word_span"][1]]["word"]))
    return found.group(0) if found else ""


def nbest_variants(mentions: list[dict], records: list[dict], words: list[dict]) -> dict[str, str]:
    # losing n-best members at each occurrence's span (spec 6.8, --expand-nbest): align every
    # alternative hypothesis of the chunk to its 1-best and read off what it put in the span
    found = defaultdict(Counter)
    cache = {}
    for m in mentions:
        rec = records[m["chunk_id"]]
        a, b = m["word_span"]
        local = [words[a]["idx"], words[b]["idx"]]
        for hyp in rec["hyps"][1:]:
            key = (m["chunk_id"], hyp["rank"])
            if key not in cache:
                cache[key] = align_doc(rec["hyps"][0]["text"], hyp["text"])
            r = realize(cache[key], local)
            if r is None:
                continue
            # an alternative that dropped or added a word is not a spelling of this span: a
            # truncation leaves audio unexplained that focus-only scoring never charges for
            surface = possessive_re.sub("", r.surface)
            norm = norm_of(surface)
            if norm and len(surface.split()) == b - a + 1:
                found[norm][surface] += 1
    return {n: c.most_common(1)[0][0] for n, c in found.items()}


def window(mention: dict, chunk_words: list[int], words: list[dict], chunk: dict, context_s: float, pad_s: float) -> tuple[float, float, list[int]]:
    # audio [s - W, e + W] with the pad on both ends, clamped to the chunk; text is every stitched
    # word of that chunk whose midpoint lies inside
    s = max(chunk["start"], mention["start"] - context_s - pad_s)
    e = min(chunk["end"], mention["end"] + context_s + pad_s)
    idx = [i for i in chunk_words if s <= (words[i]["start"] + words[i]["end"]) / 2 <= e]
    return s, e, idx


def splice(words: list[dict], idx: list[int], span: list[int], variant: str, masked: list[list[int]], suffix: str = "") -> tuple[str, tuple[int, int]]:
    # window text with the occurrence's span replaced by the variant (keeping the span's edge
    # punctuation and its possessive suffix) and co-occurring mentions of the same cluster
    # removed; returns the text and the variant's character range in it (suffix excluded)
    a, b = span
    hidden = {i for x, y in masked for i in range(x, y + 1)}
    before = [words[i]["word"] for i in idx if i < a and i not in hidden]
    after = [words[i]["word"] for i in idx if i > b and i not in hidden]
    lead = edge_re.match(words[a]["word"]).group(1)
    trail = edge_re.match(words[b]["word"]).group(3)
    head = " ".join(before)
    lo = len(head) + (1 if head else 0) + len(lead)
    text = " ".join(x for x in [head, lead + variant + suffix + trail, " ".join(after)] if x)
    return text, (lo, lo + len(variant))


def normalized(sum_lp: float, n_tok: int, length_norm: str) -> float:
    if length_norm == "mean_token":
        return sum_lp / n_tok
    if length_norm == "sum":
        return sum_lp
    if length_norm.startswith("penalty:"):
        return sum_lp / n_tok ** float(length_norm.split(":", 1)[1])
    raise ValueError(f"unknown length norm {length_norm!r}: expected mean_token, sum, or penalty:<alpha>")


def select(per_occ: dict[str, list[list]], own: list[str], conf: list[float], length_norm: str, aggregate: str) -> dict:
    # per_occ[variant][i] = [focus logprob sum, focus token count] for occurrence i; own[i] is the
    # occurrence's pass-1 spelling, whose score is the null reference s_0(i) (spec 5.8)
    scores = {v: [normalized(r[0], r[1], length_norm) for r in rows] for v, rows in per_occ.items()}
    n = len(own)
    if aggregate == "sum":
        totals = {v: sum(x) for v, x in scores.items()}
    elif aggregate == "top1":
        star = max(range(n), key=lambda i: conf[i])
        totals = {v: x[star] for v, x in scores.items()}
    elif aggregate == "median":
        # robust to one occurrence whose window supports no variant (excision or timestamp failure)
        totals = {v: statistics.median(x) * n for v, x in scores.items()}
    elif aggregate == "wins":
        # per-occurrence voting on the acoustic scores, ties broken by the summed score
        totals = {v: sum(x[i] == max(y[i] for y in scores.values()) for i in range(n)) + sum(x) / (n * 1000.0) for v, x in scores.items()}
    elif aggregate.startswith("clip:"):
        # per-occurrence advantage over the occurrence's own pass-1 spelling clipped to +-c
        c = float(aggregate.split(":", 1)[1])
        totals = {v: sum(max(-c, min(c, x[i] - scores[own[i]][i])) for i in range(n)) for v, x in scores.items()}
    else:
        raise ValueError(f"unknown aggregate {aggregate!r}: expected sum, top1, median, wins, or clip:<c>")
    ranked = sorted(totals, key=lambda v: (-totals[v], v))
    best = ranked[0]
    margin = totals[best] - totals[ranked[1]] if len(ranked) > 1 else None
    delta_null = sum(scores[best][i] - scores[own[i]][i] for i in range(len(own))) / len(own)
    return {"canonical_norm": best, "totals": totals, "margin": margin,
            "margin_per_occ": margin / len(own) if margin is not None else None, "delta_null": delta_null}


def engine_scorer(engine, audio, s: float, e: float, texts: list[str], focuses: list[tuple[int, int]]) -> list[tuple[float, int, float, int]]:
    return [(r["sum_focus"], r["n_focus_tokens"], r["sum_all"], r["n_tokens"]) for r in engine.score_texts(slice_audio(audio, s, e), texts, focuses)]


def canonicalize_doc(doc_id: str, clusters: dict, cands: list[dict], words: list[dict], records: list[dict],
                     scorer: Callable | None, params: dict) -> dict:
    by_id = {c["occ_id"]: c for c in cands}
    chunk_index = defaultdict(list)
    for i, w in enumerate(words):
        chunk_index[w["chunk_id"]].append(i)
    chunks = {r["chunk_id"]: r for r in records}
    entries, cost = [], Counter()
    started = time.monotonic()
    for k, cl in enumerate(clusters["clusters"]):
        if params["max_clusters"] and k >= params["max_clusters"]:
            break
        mentions = [by_id[o] for o in cl["occ_ids"]]
        vote_norm, weights = vote(mentions)
        pool = surfaces_by_norm(mentions)
        observed = sorted(pool)
        expanded = {}
        if params["expand_nbest"]:
            expanded = {n: s for n, s in nbest_variants(mentions, records, words).items() if n not in pool}
            pool.update(expanded)
        entry = {"cluster_id": cl["cluster_id"], "n_occ": len(mentions), "occ_ids": cl["occ_ids"],
                 "vote": pool[vote_norm], "vote_norm": vote_norm, "vote_weights": {pool[n]: w for n, w in weights.items()},
                 "variants": [pool[n] for n in observed], "expanded_variants": [pool[n] for n in sorted(expanded)],
                 "conf": [m["conf"] for m in mentions]}
        if scorer is None or len(pool) < 2:
            # canonicalizer (a), or nothing to arbitrate: the lexicon carries the vote
            ranked = sorted(weights, key=lambda n: (-weights[n], n))
            margin = weights[ranked[0]] - weights[ranked[1]] if len(ranked) > 1 else None
            entry.update({"canonical": pool[vote_norm], "canonical_norm": vote_norm, "changed_from_majority": False,
                          "scores": None, "margin": margin, "margin_per_occ": margin / len(mentions) if margin is not None else None,
                          "delta_null": None, "canonical_top1": None, "per_occ": None, "scored": False})
            entries.append(entry)
            continue

        # canonicalizer (b): every variant against every occurrence's audio, in context
        spans = [m["word_span"] for m in mentions]
        norms = sorted(pool)
        per_occ = {n: [] for n in norms}
        for m, span in zip(mentions, spans, strict=True):
            s, e, idx = window(m, chunk_index[m["chunk_id"]], words, chunks[m["chunk_id"]], params["context_window_s"], params["pad_s"])
            masked = [sp for sp in spans if sp != span and idx and idx[0] <= sp[0] and sp[1] <= idx[-1]] if params["mask_cooccurring"] else []
            suffix = possessive_of(m, words)
            texts, focuses = zip(*[splice(words, idx, span, pool[n], masked, suffix) for n in norms], strict=True)
            # rows are [focus sum, focus tokens, window sum, window tokens]; a stub scorer may give two
            for n, row in zip(norms, scorer(s, e, list(texts), list(focuses)), strict=True):
                per_occ[n].append(list(row))
            cost["n_windows"] += 1
            cost["n_forwards"] += len(norms)
            cost["scored_audio_s"] += e - s
        own = [m["norm"] for m in mentions]
        chosen = select(per_occ, own, entry["conf"], params["length_norm"], params["aggregate"])
        top1 = select(per_occ, own, entry["conf"], params["length_norm"], "top1")
        entry.update({"canonical": pool[chosen["canonical_norm"]], "canonical_norm": chosen["canonical_norm"],
                      "changed_from_majority": chosen["canonical_norm"] != vote_norm,
                      "scores": {pool[n]: t for n, t in chosen["totals"].items()}, "margin": chosen["margin"],
                      "margin_per_occ": chosen["margin_per_occ"], "delta_null": chosen["delta_null"],
                      "canonical_top1": pool[top1["canonical_norm"]], "per_occ": {pool[n]: rows for n, rows in per_occ.items()}, "scored": True})
        entries.append(entry)
    meta = {"n_clusters": len(entries), "n_scored_clusters": sum(e["scored"] for e in entries), **cost, "wall_s": round(time.monotonic() - started, 2)}
    return {"doc_id": doc_id, "canonicalizer": "b" if scorer is not None else "a", "params": params, "meta": meta, "entries": entries}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="S3: canonicalize every cluster of a run into <run-dir>/<out-subdir>/<doc>.json; (a) is the confidence-weighted vote on CPU, (b) teacher-forces every variant into every occurrence's context window and sums the length-normalized focus logprobs (spec 07 sections 5.8, 6.8). (b) stores raw per-occurrence sums so length norms, pool restriction and aggregation replay on CPU in eval_lexicon.py.")
    parser.add_argument("--run-dir", type=str, required=True, help="Pass-1 run dir with candidates and clusters (required).")
    parser.add_argument("--canonicalizer", type=str, default="b", choices=["a", "b"], help="a = vote, b = cross-occurrence acoustic rescoring (default: b).")
    parser.add_argument("--model", type=str, default="large-v3", help="openai-whisper model for (b); must match the pass-1 model for delta_null to mean anything (default: large-v3).")
    parser.add_argument("--manifest", type=str, default="data/derived/earnings21-conec/manifest.jsonl", help="Corpus manifest for audio paths and split membership (default: data/derived/earnings21-conec/manifest.jsonl).")
    parser.add_argument("--split", type=str, default="all", help="Split to process, or all (default: all).")
    parser.add_argument("--docs", type=str, default="", help="Comma-separated doc ids overriding the split (default: manifest split).")
    parser.add_argument("--clusters-subdir", type=str, default="clusters", help="Clusters subdir under the run dir (default: clusters).")
    parser.add_argument("--candidates-subdir", type=str, default="candidates", help="Candidates subdir under the run dir (default: candidates).")
    parser.add_argument("--out-subdir", type=str, default="", help="Output subdir under the run dir; empty means lexicon_<canonicalizer> (default: derived).")
    parser.add_argument("--context-window-s", type=float, default=5.0, help="Context W on each side of the occurrence for (b) (default: 5.0).")
    parser.add_argument("--pad-s", type=float, default=0.5, help="Extra audio pad on the window edges for timestamp noise (default: 0.5).")
    parser.add_argument("--length-norm", type=str, default="mean_token", help="mean_token, sum, or penalty:<alpha> for the recorded canonical (default: mean_token).")
    parser.add_argument("--aggregate", type=str, default="sum", help="sum (the method), top1 (highest-confidence occurrence only), median, wins, or clip:<c> (default: sum).")
    parser.add_argument("--expand-nbest", action="store_true", help="Add losing n-best spellings at each occurrence to the variant pool (default: False).")
    parser.add_argument("--no-mask-cooccurring", action="store_true", help="Leave other mentions of the same cluster inside the window text; the ablation (default: False).")
    parser.add_argument("--max-clusters", type=int, default=0, help="Score at most this many clusters per doc, for smokes; 0 means all (default: 0).")
    parser.add_argument("--force", action="store_true", help="Rewrite docs whose lexicon already exists (default: False).")
    args = parser.parse_args()

    run = Path(args.run_dir)
    out_subdir = args.out_subdir or f"lexicon_{args.canonicalizer}"
    manifest = {m["doc_id"]: m for m in read_jsonl(args.manifest)}
    docs = args.docs.split(",") if args.docs else sorted(d for d, m in manifest.items() if args.split in (m["split"], "all"))
    if not docs:
        raise ValueError(f"no docs in split {args.split!r} of {args.manifest}")
    os.makedirs(run / out_subdir, exist_ok=True)
    params = {"context_window_s": args.context_window_s, "pad_s": args.pad_s, "length_norm": args.length_norm, "aggregate": args.aggregate,
              "expand_nbest": args.expand_nbest, "mask_cooccurring": not args.no_mask_cooccurring, "max_clusters": args.max_clusters,
              "model": args.model if args.canonicalizer == "b" else None}

    engine = None
    if args.canonicalizer == "b":
        from scripts.common.whisper_engine import WhisperEngine

        engine = WhisperEngine(args.model)
    totals, started = Counter(), time.monotonic()
    for doc in tqdm(docs, desc=f"canonicalize {args.canonicalizer}"):
        out_path = run / out_subdir / f"{doc}.json"
        if out_path.exists() and not args.force:
            continue
        clusters = json.loads((run / args.clusters_subdir / f"{doc}.json").read_text(encoding="utf-8"))
        cands = read_jsonl(run / args.candidates_subdir / f"{doc}.jsonl")
        words = stitched_words(run, doc)
        records = read_jsonl(run / "pass1" / f"{doc}.jsonl")
        scorer = partial(engine_scorer, engine, load_audio(manifest[doc]["audio_path"])) if engine is not None else None
        result = canonicalize_doc(doc, clusters, cands, words, records, scorer, params)
        out_path.write_text(json.dumps(result, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
        totals.update({k: v for k, v in result["meta"].items() if k != "wall_s"})
        totals["docs"] += 1
        totals["doc_audio_s"] += manifest[doc]["duration_s"]
        totals["changed_from_majority"] += sum(e["changed_from_majority"] for e in result["entries"])

    wall = time.monotonic() - started
    stage = f"canonicalize_{out_subdir}"
    append_config(run, stage, {"argv": vars(args), "params": params, **totals, "wall_s": round(wall, 1),
                               "s3_rtf": round(wall / totals["doc_audio_s"], 5) if totals["doc_audio_s"] else None,
                               "device": str(engine.device) if engine else "cpu", "started_utc": datetime.now(UTC).isoformat(timespec="seconds")})
    print(f"{totals['docs']} docs -> {run / out_subdir}: {totals['n_clusters']} entries, {totals['n_scored_clusters']} scored "
          f"({totals['n_windows']} windows, {totals['n_forwards']} forwards), changed_from_majority {totals['changed_from_majority']}, "
          f"wall {wall:.0f} s, S3 RTF {wall / totals['doc_audio_s'] if totals['doc_audio_s'] else 0:.4f}")
