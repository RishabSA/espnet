import json
from pathlib import Path

import pytest

from scripts.analysis.e1_windows import stitched_words
from scripts.canon.canonicalize import (
    canonicalize_doc,
    nbest_variants,
    normalized,
    select,
    splice,
    vote,
)
from scripts.canon.eval_lexicon import doc_lexicon_metrics
from scripts.common.align import align_doc
from scripts.common.io import read_hyp_text, read_jsonl
from scripts.decode.pass2 import apply_a0
from scripts.mine.cluster_variants import build_clusters
from scripts.mine.mine_candidates import load_stoplist, mine_doc

fixture_dir = Path("tests/fixtures/tiny_doc")
params = {"context_window_s": 5.0, "pad_s": 0.5, "length_norm": "mean_token", "aggregate": "sum", "expand_nbest": False, "mask_cooccurring": True, "max_clusters": 0}

# docs/10 sections 4 and 5: seven occurrences of one entity, confidences and the 21 scorings
confs = [-0.42, -0.71, -0.55, -0.95, -0.68, -0.38, -0.80]
surfaces = ["Kowalski", "Kowalsky", "Kowalsky", "Covalski", "Kowalsky", "Kowalski", "Kowalsky"]
table = {
    "Kowalski": [-0.31, -0.52, -0.44, -0.61, -0.49, -0.28, -0.55],
    "Kowalsky": [-0.38, -0.47, -0.41, -0.72, -0.45, -0.36, -0.51],
    "Covalski": [-0.55, -0.68, -0.63, -0.58, -0.71, -0.60, -0.69],
}


def _mentions() -> list[dict]:
    return [{"occ_id": f"d#c0#w{10 * i:04d}", "chunk_id": 0, "word_span": [10 * i, 10 * i], "surface": s, "norm": s.lower(),
             "start": 10.0 * i + 1, "end": 10.0 * i + 1.5, "conf": c} for i, (s, c) in enumerate(zip(surfaces, confs, strict=True))]


def _fixture_cands(doc: str) -> list[dict]:
    return mine_doc(doc, stitched_words(fixture_dir, doc), [], {"rarity", "caps", "lowconf"}, 3.5, -0.8, 4, load_stoplist("scripts/mine/stoplist_earnings.txt"))


def test_vote_picks_the_majority_and_is_wrong():
    norm, weights = vote(_mentions())
    assert norm == "kowalsky"
    assert weights["kowalsky"] == pytest.approx(2.025, abs=2e-3)
    assert weights["kowalski"] == pytest.approx(1.341, abs=2e-3) and weights["covalski"] == pytest.approx(0.387, abs=2e-3)


def test_select_reproduces_the_worked_example():
    per_occ = {v.lower(): [[x * 4, 4] for x in rows] for v, rows in table.items()}
    own = [s.lower() for s in surfaces]
    r = select(per_occ, own, confs, "mean_token", "sum")
    assert r["canonical_norm"] == "kowalski"
    assert r["totals"]["kowalski"] == pytest.approx(-3.20) and r["totals"]["kowalsky"] == pytest.approx(-3.30) and r["totals"]["covalski"] == pytest.approx(-4.44)
    assert r["margin"] == pytest.approx(0.10) and r["margin_per_occ"] == pytest.approx(0.10 / 7)
    # kowalski loses five of seven occurrences outright and still wins the sum
    winners = [max(table, key=lambda v: table[v][i]) for i in range(7)]
    assert winners.count("Kowalski") == 2
    # delta_null against the pass-1 spelling at each occurrence is negative here: the local
    # winners are the pass-1 spellings, so the sum-winner trails them wherever it disagrees
    assert r["delta_null"] == pytest.approx(-0.19 / 7)
    # the top-1 aggregation is the per-utterance scorer of the prior art; the most confident occurrence is 6
    assert select(per_occ, own, confs, "mean_token", "top1")["canonical_norm"] == "kowalski"
    # robust aggregations: kowalski loses the median and the win count (it wins only 2 of 7) but keeps
    # the clipped sum, and one wild occurrence cannot swing the median
    assert select(per_occ, own, confs, "mean_token", "median")["canonical_norm"] == "kowalsky"
    assert select(per_occ, own, confs, "mean_token", "wins")["canonical_norm"] == "kowalsky"
    # clipping the per-occurrence advantage at 0.05 removes the wide wins that carry kowalski;
    # a clip above the largest delta leaves the sum-of-deltas ordering intact
    assert select(per_occ, own, confs, "mean_token", "clip:0.05")["canonical_norm"] == "kowalsky"
    assert select(per_occ, own, confs, "mean_token", "clip:0.5")["canonical_norm"] == "kowalski"
    wild = {v: [list(r) for r in rows] for v, rows in per_occ.items()}
    wild["covalski"][3] = [-0.01 * 4, 4]
    assert select(wild, own, confs, "mean_token", "sum")["canonical_norm"] == "kowalski"
    wild["covalski"][3] = [8.0 * 4, 4]
    assert select(wild, own, confs, "mean_token", "sum")["canonical_norm"] == "covalski"
    assert select(wild, own, confs, "mean_token", "median")["canonical_norm"] == "kowalsky"
    with pytest.raises(ValueError):
        select(per_occ, own, confs, "mean_token", "bogus")


def test_length_norm_confound():
    assert normalized(-2.0, 4, "mean_token") == -0.5 and normalized(-2.0, 4, "sum") == -2.0 and normalized(-2.0, 4, "penalty:0.5") == pytest.approx(-1.0)
    with pytest.raises(ValueError):
        normalized(-2.0, 4, "bogus")
    # docs/10 occurrence 2: on raw sums the correct spelling ranks last because Covalski has one token fewer
    raw = {"kowalski": [[-2.08, 4]], "kowalsky": [[-1.88, 4]], "covalski": [[-2.04, 3]]}
    by_sum = select(raw, ["kowalsky"], [-0.7], "sum", "sum")["totals"]
    by_mean = select(raw, ["kowalsky"], [-0.7], "mean_token", "sum")["totals"]
    assert sorted(raw, key=lambda v: -by_sum[v]) == ["kowalsky", "covalski", "kowalski"]
    assert sorted(raw, key=lambda v: -by_mean[v]) == ["kowalsky", "kowalski", "covalski"]


def test_splice_keeps_punctuation_and_masks_cooccurring():
    words = [{"word": w, "start": float(i), "end": i + 0.5, "chunk_id": 0} for i, w in enumerate(["turn", "it", "over", "to", "Kowalsky,", "our", "chief", "executive", "Kowalski", "to", "walk"])]
    idx = list(range(len(words)))
    text, focus = splice(words, idx, [4, 4], "Kowalski", [])
    assert text == "turn it over to Kowalski, our chief executive Kowalski to walk" and text[focus[0] : focus[1]] == "Kowalski"
    text, focus = splice(words, idx, [4, 4], "Covalski", [[8, 8]])
    assert text == "turn it over to Covalski, our chief executive to walk" and text[focus[0] : focus[1]] == "Covalski"
    words = [{"word": w, "start": float(i), "end": i + 0.5, "chunk_id": 0} for i, w in enumerate(["met", "(Kai", "Fu", "Lee),", "yesterday"])]
    text, focus = splice(words, list(range(5)), [1, 3], "Kaifu Lee", [])
    assert text == "met (Kaifu Lee), yesterday" and text[focus[0] : focus[1]] == "Kaifu Lee"
    text, focus = splice(words, list(range(5)), [0, 0], "Met", [])
    assert text == "Met (Kai Fu Lee), yesterday" and focus == (0, 3)
    # the occurrence's possessive is re-attached outside the focus, so "Nexstar's" and "Nextar"
    # are scored with the same morphology the audio carries
    words = [{"word": w, "start": float(i), "end": i + 0.5, "chunk_id": 0} for i, w in enumerate(["we", "saw", "Nexstar's,", "results"])]
    text, focus = splice(words, list(range(4)), [2, 2], "Nextar", [], "'s")
    assert text == "we saw Nextar's, results" and text[focus[0] : focus[1]] == "Nextar"


def test_nbest_variants():
    words = [{"word": w, "start": float(i), "end": i + 0.5, "chunk_id": 0, "idx": i} for i, w in enumerate(["we", "met", "Pemsa", "today"])]
    records = [{"doc_id": "d", "chunk_id": 0, "start": 0.0, "end": 4.0, "hyps": [
        {"rank": 0, "text": "we met Pemsa today"}, {"rank": 1, "text": "we met Femsa today"}, {"rank": 2, "text": "we met Pemsa today."}, {"rank": 3, "text": "we met today"},
        {"rank": 4, "text": "we met Pemsa Corp today"}, {"rank": 5, "text": "we met Femsa's today"}]}]
    m = {"occ_id": "d#c0#w0002", "chunk_id": 0, "word_span": [2, 2], "surface": "Pemsa", "norm": "pemsa", "start": 2.0, "end": 2.5, "conf": -0.5}
    # a two-word alternative is not a spelling of a one-word span; possessives fold onto the base
    assert nbest_variants([m], records, words) == {"pemsa": "Pemsa", "femsa": "Femsa"}


def _stub_scorer(s: float, e: float, texts: list[str], focuses: list[tuple[int, int]]) -> list[tuple[float, int]]:
    # reads the spliced variant back out of the focus and looks the score up by window position
    occ = round((s + e) / 2 / 10)
    return [(table[text[lo:hi]][occ] * 4, 4) for text, (lo, hi) in zip(texts, focuses, strict=True)]


def _synthetic_doc() -> tuple[dict, list[dict], list[dict], list[dict]]:
    mentions = _mentions()
    words = [{"word": f"w{i}", "start": float(i), "end": i + 0.5, "chunk_id": 0, "idx": i} for i in range(70)]
    for m in mentions:
        words[m["word_span"][0]]["word"] = m["surface"]
    records = [{"doc_id": "d", "chunk_id": 0, "start": 0.0, "end": 70.0, "hyps": [{"rank": 0, "text": " ".join(w["word"] for w in words)}]}]
    clusters = {"doc_id": "d", "clusters": [{"cluster_id": "c_000", "occ_ids": [m["occ_id"] for m in mentions]}]}
    return clusters, mentions, words, records


def test_canonicalize_doc_b_recovers_kowalski_and_a_carries_the_vote():
    clusters, mentions, words, records = _synthetic_doc()
    lex = canonicalize_doc("d", clusters, mentions, words, records, _stub_scorer, params)
    e = lex["entries"][0]
    assert e["vote"] == "Kowalsky" and e["canonical"] == "Kowalski" and e["changed_from_majority"] and e["scored"]
    assert e["scores"]["Kowalski"] == pytest.approx(-3.20) and e["margin_per_occ"] == pytest.approx(0.10 / 7) and e["canonical_top1"] == "Kowalski"
    assert sorted(e["variants"]) == ["Covalski", "Kowalski", "Kowalsky"] and e["expanded_variants"] == []
    assert lex["meta"] == {**lex["meta"], "n_clusters": 1, "n_scored_clusters": 1, "n_windows": 7, "n_forwards": 21}
    assert lex["canonicalizer"] == "b" and len(e["per_occ"]["Kowalski"]) == 7
    lex_a = canonicalize_doc("d", clusters, mentions, words, records, None, params)
    a = lex_a["entries"][0]
    assert lex_a["canonicalizer"] == "a" and a["canonical"] == "Kowalsky" and a["scores"] is None and not a["changed_from_majority"] and not a["scored"]


def test_a0_reproduces_fixture_pass2_and_empty_lexicon_is_identity():
    records = read_jsonl(fixture_dir / "pass1" / "d1.jsonl")
    word_records = read_jsonl(fixture_dir / "pass1" / "d1.words.jsonl")
    cands = {c["occ_id"]: c for c in _fixture_cands("d1")}
    entry = {"cluster_id": "c_000", "canonical": "Kowalski", "canonical_norm": "kowalski", "occ_ids": ["d1#c0#w0012", "d1#c0#w0019", "d1#c0#w0029"]}
    out, text = apply_a0(records, word_records, cands, [entry])
    expected = read_jsonl(fixture_dir / "pass2" / "d1.jsonl")[0]["hyps"][0]["text"]
    assert text == expected and out[0]["hyps"][0]["text"] == expected and out[0]["variant"] == "a0"
    assert [(e["from"], e["to"], e["occ_id"]) for e in out[0]["edits"]] == [("Kowalsky", "Kowalski", "d1#c0#w0012"), ("Kowalsky", "Kowalski", "d1#c0#w0029")]
    same, same_text = apply_a0(records, word_records, cands, [])
    assert same_text == records[0]["hyps"][0]["text"] and same[0]["edits"] == []
    # a possessive occurrence keeps its possessive when the bare canonical is substituted
    word_records2 = [{"chunk_id": 0, "words": [{"word": w, "stitched": True} for w in ["we", "saw", "Nexstar's,", "and", "Nextar", "grew"]]}]
    records2 = [{"doc_id": "x", "chunk_id": 0, "start": 0.0, "end": 3.0, "hyps": [{"rank": 0, "text": "we saw Nexstar's, and Nextar grew"}]}]
    cands2 = {"x#c0#w0002": {"word_span": [2, 2], "surface": "Nexstar's"}, "x#c0#w0004": {"word_span": [4, 4], "surface": "Nextar"}}
    out2, text2 = apply_a0(records2, word_records2, cands2, [{"cluster_id": "c", "canonical": "Nexstar", "canonical_norm": "nexstar", "occ_ids": ["x#c0#w0002", "x#c0#w0004"]}])
    assert text2 == "we saw Nexstar's, and Nexstar grew" and [e["occ_id"] for e in out2[0]["edits"]] == ["x#c0#w0004"]


def test_eval_lexicon_scores_the_vote_on_the_fixture():
    doc = "d1"
    cands = _fixture_cands(doc)
    clusters = build_clusters(doc, cands, 0.25, 0.10, 2, 0.25, 3.5)
    lex = canonicalize_doc(doc, clusters, cands, stitched_words(fixture_dir, doc), read_jsonl(fixture_dir / "pass1" / "d1.jsonl"), None, params)
    entities = json.loads((fixture_dir / "ref_entities" / "d1.json").read_text())["entities"]
    alignment = align_doc((fixture_dir / "refs" / "d1.txt").read_text(), read_hyp_text(fixture_dir, "pass1", doc))
    c, rows = doc_lexicon_metrics(doc, lex, cands, alignment, entities, None, False, None)
    row = next(r for r in rows if r["n_occ"] == 3)
    assert row["vote"] == "kowalsky" and row["truth"] == "kowalski" and row["oracle"] == "kowalski" and not row["choice_correct"]
    assert (row["canonical_damage"], row["canonical_repair"], row["oracle_damage"], row["oracle_repair"]) == (1, 0, 0, 2)
    assert c["multi"] == 1 and c["multi_choice_correct"] == 0 and c["multi_oracle_correct"] == 1


@pytest.mark.slow
def test_score_texts_batching_matches_single_scoring():
    import torch

    from scripts.common.audio import load_audio
    from scripts.common.whisper_engine import WhisperEngine

    engine = WhisperEngine("tiny", device=torch.device("cpu"))
    audio = load_audio("tests/fixtures/audio/say_sample.wav")
    hyp = engine.n_best_decode(audio, num_beams=2, num_return_sequences=1)[0]["text"]
    texts = [hyp, hyp + " and then some more words follow here"]
    focuses = [(0, min(6, len(hyp))), (0, min(6, len(hyp)))]
    batch = engine.score_texts(audio, texts, focuses)
    single = [engine.score_text(audio, t, f) for t, f in zip(texts, focuses, strict=True)]
    for b, s in zip(batch, single, strict=True):
        assert b["n_tokens"] == s["n_tokens"] and b["n_focus_tokens"] == s["n_focus_tokens"]
        assert b["sum_all"] == pytest.approx(s["sum_all"], abs=1e-2) and b["sum_focus"] == pytest.approx(s["sum_focus"], abs=1e-2)
