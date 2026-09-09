import argparse
import json
import os
from collections import Counter
from pathlib import Path

from scripts.canon.canonicalize import norm_of, select
from scripts.common.align import align_doc
from scripts.common.io import append_config, git_sha, read_hyp_text, read_jsonl, write_jsonl
from scripts.eval.evaluate import alignable_entities
from scripts.mine.eval_clusters import hyp_word_to_ref, mention_labels, ref_word_entities

# calibration thresholds on delta_null (spec 5.8: gates read the null-referenced margin)
delta_thresholds = [None, -0.10, -0.05, -0.02, 0.0, 0.02, 0.05]


def replay(entry: dict, length_norm: str | None, observed_only: bool, aggregate: str | None) -> tuple[str, float | None]:
    # recompute the canonical from the stored per-occurrence sums under other settings; the
    # lexicon's recorded choice stands when nothing is overridden or the entry was not scored
    if entry["per_occ"] is None or (length_norm is None and not observed_only and aggregate is None):
        return entry["canonical_norm"], entry["delta_null"]
    per_occ = {norm_of(v): rows for v, rows in entry["per_occ"].items()}
    if observed_only:
        keep = {norm_of(v) for v in entry["variants"]}
        per_occ = {n: rows for n, rows in per_occ.items() if n in keep}
    chosen = select(per_occ, entry["own_norms"], entry["conf"], length_norm or "mean_token", aggregate or "sum")
    return chosen["canonical_norm"], chosen["delta_null"]


def doc_lexicon_metrics(doc_id: str, lexicon: dict, cands: list[dict], alignment, entities: list[dict], length_norm: str | None,
                        observed_only: bool, aggregate: str | None) -> tuple[Counter, list[dict]]:
    entities, _ = alignable_entities(alignment, entities)
    word_ref = hyp_word_to_ref(alignment)
    by_word, named = ref_word_entities(entities)
    by_id = {c["occ_id"]: c for c in cands}
    c, rows = Counter(), []
    for entry in lexicon["entries"]:
        mentions = [by_id[o] for o in entry["occ_ids"]]
        entry["own_norms"] = [m["norm"] for m in mentions]
        labels = [mention_labels(m, alignment, word_ref, by_word, named) for m in mentions]
        labeled = [(m["norm"].replace(" ", ""), sp) for m, (sp, _, _) in zip(mentions, labels, strict=True) if sp is not None]
        if not labeled or not any(l[2] for l in labels):
            continue
        canonical_norm, delta_null = replay(entry, length_norm, observed_only, aggregate)
        choice = canonical_norm.replace(" ", "")
        truth = Counter(sp for _, sp in labeled).most_common(1)[0][0]
        pool = sorted({n for n, _ in labeled} | {norm_of(v).replace(" ", "") for v in (entry["per_occ"] or entry["variants"])})
        oracle = max(pool, key=lambda v: (sum(v == sp for _, sp in labeled), v))
        vote_norm = entry["vote_norm"].replace(" ", "")
        multi = len(pool) >= 2
        row = {"doc_id": doc_id, "cluster_id": entry["cluster_id"], "n_occ": entry["n_occ"], "n_variants": len(pool), "scored": entry["scored"],
               "canonical": canonical_norm, "vote": entry["vote_norm"], "truth": truth, "oracle": oracle,
               "choice_correct": choice == truth, "vote_correct": vote_norm == truth, "matches_oracle": choice == oracle,
               "changed_from_majority": choice != vote_norm, "delta_null": delta_null, "margin_per_occ": entry["margin_per_occ"]}
        for tag, pick in (("canonical", choice), ("vote", vote_norm), ("oracle", oracle)):
            dmg = sum(n == sp and pick != sp for n, sp in labeled)
            rep = sum(n != sp and pick == sp for n, sp in labeled)
            row[f"{tag}_damage"], row[f"{tag}_repair"] = dmg, rep
            c[f"{tag}_damage"] += dmg
            c[f"{tag}_repair"] += rep
        c["labeled"] += len(labeled)
        c["correct_now"] += sum(n == sp for n, sp in labeled)
        c["entries"] += 1
        if multi:
            c["multi"] += 1
            c["multi_choice_correct"] += row["choice_correct"]
            c["multi_vote_correct"] += row["vote_correct"]
            c["multi_oracle_correct"] += oracle == truth
            c["multi_matches_oracle"] += row["matches_oracle"]
        if row["changed_from_majority"]:
            c["changed"] += 1
            c["changed_right"] += row["choice_correct"] and not row["vote_correct"]
            c["changed_wrong"] += row["vote_correct"] and not row["choice_correct"]
        rows.append(row)
    return c, rows


def ratio(num: float, den: float) -> float | None:
    return num / den if den else None


def calibration(rows: list[dict]) -> list[dict]:
    # enforce only clusters whose delta_null clears t: the A2 gate-4 curve, replayed from the lexicon
    out = []
    scored = [r for r in rows if r["scored"] and r["delta_null"] is not None]
    for t in delta_thresholds:
        kept = scored if t is None else [r for r in scored if r["delta_null"] >= t]
        multi = [r for r in kept if r["n_variants"] >= 2]
        out.append({"delta_min": t, "n_clusters": len(kept), "choice_accuracy": ratio(sum(r["choice_correct"] for r in multi), len(multi)),
                    "damage": sum(r["canonical_damage"] for r in kept), "repair": sum(r["canonical_repair"] for r in kept)})
    return out


def summarize(c: Counter, rows: list[dict]) -> dict:
    return {
        "n_entries": c["entries"], "n_multi_variant": c["multi"], "labeled_mentions": c["labeled"], "correct_now": ratio(c["correct_now"], c["labeled"]),
        "choice_accuracy": ratio(c["multi_choice_correct"], c["multi"]), "vote_accuracy": ratio(c["multi_vote_correct"], c["multi"]),
        "oracle_accuracy": ratio(c["multi_oracle_correct"], c["multi"]), "matches_oracle": ratio(c["multi_matches_oracle"], c["multi"]),
        "changed_from_majority": ratio(c["changed"], c["entries"]), "changed_right": c["changed_right"], "changed_wrong": c["changed_wrong"],
        **{f"{t}_{k}_rate": ratio(c[f"{t}_{k}"], c["labeled"]) for t in ("canonical", "vote", "oracle") for k in ("damage", "repair")},
        **{f"{t}_net": ratio(c[f"{t}_repair"] - c[f"{t}_damage"], c["labeled"]) for t in ("canonical", "vote", "oracle")},
        "calibration": calibration(rows), "totals": dict(c),
    }


def evaluate(run_dir: str | Path, lexicon_subdir: str, cand_subdir: str, refs_dir: str | Path, ref_entities_dir: str | Path, doc_ids: list[str],
             length_norm: str | None = None, observed_only: bool = False, aggregate: str | None = None) -> dict:
    run = Path(run_dir)
    total, rows = Counter(), []
    for doc in doc_ids:
        lexicon = json.loads((run / lexicon_subdir / f"{doc}.json").read_text(encoding="utf-8"))
        cands = read_jsonl(run / cand_subdir / f"{doc}.jsonl")
        entities = json.loads((Path(ref_entities_dir) / f"{doc}.json").read_text(encoding="utf-8"))["entities"]
        alignment = align_doc((Path(refs_dir) / f"{doc}.txt").read_text(encoding="utf-8"), read_hyp_text(run, "pass1", doc))
        c, r = doc_lexicon_metrics(doc, lexicon, cands, alignment, entities, length_norm, observed_only, aggregate)
        total.update(c)
        rows.extend(r)
    return {"summary": summarize(total, rows), "clusters": rows}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="M5.5: score a lexicon against the reference: canonical-choice accuracy vs the vote and the pool-restricted oracle on multi-variant named clusters, the damage/repair each choice would inflict, changed_from_majority right/wrong, and the delta_null calibration curve; length norm, pool restriction and aggregation replay from the stored scores without the GPU.")
    parser.add_argument("--run-dir", type=str, required=True, help="Pass-1 run dir holding the lexicon (required).")
    parser.add_argument("--lexicon-subdir", type=str, default="lexicon_b", help="Lexicon subdir under the run dir (default: lexicon_b).")
    parser.add_argument("--candidates-subdir", type=str, default="candidates", help="Candidates subdir under the run dir (default: candidates).")
    parser.add_argument("--refs-dir", type=str, default="data/derived/earnings21-conec/refs", help="Reference transcripts (default: data/derived/earnings21-conec/refs).")
    parser.add_argument("--ref-entities", type=str, default="data/derived/earnings21-conec/ref_entities", help="Reference entity index dir (default: data/derived/earnings21-conec/ref_entities).")
    parser.add_argument("--manifest", type=str, default="data/derived/earnings21-conec/manifest.jsonl", help="Corpus manifest for split membership (default: data/derived/earnings21-conec/manifest.jsonl).")
    parser.add_argument("--split", type=str, default="dev", help="Split to score, or all (default: dev).")
    parser.add_argument("--docs", type=str, default="", help="Comma-separated doc ids overriding the manifest lookup (default: manifest split).")
    parser.add_argument("--length-norm", type=str, default="", help="Replay the choice under mean_token, sum or penalty:<alpha>; empty keeps the recorded choice (default: recorded).")
    parser.add_argument("--observed-only", action="store_true", help="Replay with n-best-expanded variants removed from the pool (default: False).")
    parser.add_argument("--aggregate", type=str, default="", help="Replay with sum, top1, median, wins, or clip:<c> aggregation; empty keeps the recorded choice (default: recorded).")
    parser.add_argument("--tag", type=str, default="", help="Suffix for the metrics dir when replaying, e.g. sum or top1 (default: none).")
    args = parser.parse_args()

    docs = args.docs.split(",") if args.docs else sorted(m["doc_id"] for m in read_jsonl(args.manifest) if args.split in (m["split"], "all"))
    if not docs:
        raise ValueError(f"no docs in split {args.split!r} of {args.manifest}")
    result = evaluate(args.run_dir, args.lexicon_subdir, args.candidates_subdir, args.refs_dir, args.ref_entities, docs,
                      args.length_norm or None, args.observed_only, args.aggregate or None)
    name = f"{args.lexicon_subdir}_{args.split}" + (f"_{args.tag}" if args.tag else "")
    out = Path(args.run_dir) / "metrics" / name
    os.makedirs(out, exist_ok=True)
    sha, dirty = git_sha()
    (out / "summary.json").write_text(json.dumps({"meta": {"run_dir": args.run_dir, "split": args.split, "n_docs": len(docs), "git_sha": sha, "dirty": dirty, "argv": vars(args)},
                                                  "metrics": result["summary"]}, indent=2) + "\n", encoding="utf-8")
    write_jsonl(out / "clusters.jsonl", result["clusters"])
    append_config(args.run_dir, f"eval_{name}", {"argv": vars(args), "out": str(out)})
    s = result["summary"]

    def cell(v: float | None) -> str:
        return "-" if v is None else f"{v:.4f}"

    print(f"{len(docs)} docs, {s['n_entries']} named entries ({s['n_multi_variant']} multi-variant), correct now {cell(s['correct_now'])}")
    print(f"choice accuracy {cell(s['choice_accuracy'])} vs vote {cell(s['vote_accuracy'])} vs oracle {cell(s['oracle_accuracy'])}; matches oracle {cell(s['matches_oracle'])}; "
          f"changed_from_majority {cell(s['changed_from_majority'])} (right {s['changed_right']}, wrong {s['changed_wrong']})")
    for t in ("canonical", "vote", "oracle"):
        print(f"{t}: damage {cell(s[f'{t}_damage_rate'])} repair {cell(s[f'{t}_repair_rate'])} net {cell(s[f'{t}_net'])}")
    print("| delta_null >= | clusters | choice accuracy | damage | repair |")
    print("|---|---|---|---|---|")
    for r in s["calibration"]:
        print(f"| {'any' if r['delta_min'] is None else r['delta_min']} | {r['n_clusters']} | {cell(r['choice_accuracy'])} | {r['damage']} | {r['repair']} |")
