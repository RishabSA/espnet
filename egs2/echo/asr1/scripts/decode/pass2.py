import argparse
import json
import os
import re
from collections import defaultdict
from pathlib import Path

from tqdm import tqdm

from scripts.common.io import append_config, create_run_dir, read_jsonl, write_jsonl

edge_re = re.compile(r"^(\W*)(.*?)(\W*)$", re.DOTALL)


def apply_a0(records: list[dict], word_records: list[dict], cands: dict[str, dict], entries: list[dict]) -> tuple[list[dict], str]:
    # replace every clustered mention's span with its cluster's canonical spelling in the pass-1
    # words, keeping the span's edge punctuation; returns the pass-2 chunk records and the
    # stitched transcript (spec 07 section 6.9, A0: no audio, no gates)
    chunk_words = {w["chunk_id"]: [x["word"] for x in w["words"]] for w in word_records}
    stitched = [(w["chunk_id"], j) for w in word_records for j, x in enumerate(w["words"]) if x.get("stitched")]
    edits = defaultdict(list)
    for entry in entries:
        for occ_id in entry["occ_ids"]:
            m = cands[occ_id]
            a, b = m["word_span"]
            chunk, first = stitched[a]
            _, last = stitched[b]
            raw_first, raw_last = chunk_words[chunk][first], chunk_words[chunk][last]
            if raw_first is None or raw_last is None:
                raise ValueError(f"{occ_id}: span overlaps an already substituted mention")
            lead, trail = edge_re.match(raw_first).group(1), edge_re.match(raw_last).group(3)
            core = " ".join(x for x in chunk_words[chunk][first : last + 1] if x is not None)
            core = edge_re.match(core).group(2)
            if core == entry["canonical"]:
                continue
            chunk_words[chunk][first] = lead + entry["canonical"] + trail
            for j in range(first + 1, last + 1):
                chunk_words[chunk][j] = None
            edits[chunk].append({"occ_id": occ_id, "from": m["surface"], "to": entry["canonical"], "cluster_id": entry["cluster_id"], "reason": "a0"})

    out = []
    for r in records:
        text = " ".join(x for x in chunk_words[r["chunk_id"]] if x is not None)
        out.append({"doc_id": r["doc_id"], "chunk_id": r["chunk_id"], "start": r["start"], "end": r["end"],
                    "hyps": [{"rank": 0, "text": text}], "variant": "a0", "edits": edits.get(r["chunk_id"], [])})
    transcript = " ".join(chunk_words[c][j] for c, j in stitched if chunk_words[c][j] is not None)
    return out, transcript


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="S4: second pass from a lexicon into a new run dir's pass2/ with an edits audit trail (spec 07 sections 5.9, 6.9). Only --variant a0 (text substitution) exists so far.")
    parser.add_argument("--run-dir", type=str, required=True, help="Pass-1 run dir with candidates, clusters and the lexicon (required).")
    parser.add_argument("--lexicon-subdir", type=str, default="lexicon_b", help="Lexicon subdir under the run dir (default: lexicon_b).")
    parser.add_argument("--candidates-subdir", type=str, default="candidates", help="Candidates subdir under the run dir (default: candidates).")
    parser.add_argument("--variant", type=str, default="a0", choices=["a0"], help="Enforcement variant (default: a0).")
    parser.add_argument("--out-run", type=str, required=True, help="Run dir to write pass2/ into (required).")
    parser.add_argument("--docs", type=str, default="", help="Comma-separated doc ids; empty means every doc with a lexicon (default: all).")
    parser.add_argument("--force", action="store_true", help="Write into a non-empty out run, skipping docs already done (default: False).")
    args = parser.parse_args()

    run = Path(args.run_dir)
    out = create_run_dir(args.out_run, force=args.force)
    os.makedirs(out / "pass2", exist_ok=True)
    docs = args.docs.split(",") if args.docs else sorted(p.stem for p in (run / args.lexicon_subdir).glob("*.json"))
    if not docs:
        raise FileNotFoundError(f"no lexicon files under {run / args.lexicon_subdir}")

    n_edits = n_docs = 0
    for doc in tqdm(docs, desc=f"pass2 {args.variant}"):
        out_path = out / "pass2" / f"{doc}.jsonl"
        if out_path.exists() and not args.force:
            continue
        records = read_jsonl(run / "pass1" / f"{doc}.jsonl")
        word_records = read_jsonl(run / "pass1" / f"{doc}.words.jsonl")
        cands = {c["occ_id"]: c for c in read_jsonl(run / args.candidates_subdir / f"{doc}.jsonl")}
        lexicon = json.loads((run / args.lexicon_subdir / f"{doc}.json").read_text(encoding="utf-8"))
        out_records, transcript = apply_a0(records, word_records, cands, lexicon["entries"])
        write_jsonl(out_path, out_records)
        (out / "pass2" / f"{doc}.txt").write_text(transcript + "\n", encoding="utf-8")
        n_edits += sum(len(r["edits"]) for r in out_records)
        n_docs += 1

    append_config(out, "pass2", {"argv": vars(args), "source_run": args.run_dir, "lexicon_subdir": args.lexicon_subdir, "variant": args.variant,
                                 "docs": n_docs, "edits": n_edits})
    print(f"{n_docs} docs -> {out / 'pass2'}: {n_edits} substitutions")
