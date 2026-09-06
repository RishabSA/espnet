#!/bin/bash
#SBATCH --job-name=e21-canon-b
#SBATCH --partition=gpuA40x4
#SBATCH --account=bbjs-delta-gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=60000M
#SBATCH --time=12:00:00
#SBATCH --output=slurm-%x-%j.out

# M5.2 / M5.4 / M5.7: canonicalizer (b) on every Earnings-21 document (dev + test) with the n-best
# expanded pool, the masking ablation as a second scoring run, then the CPU replays and the A0
# transition matrices. canonicalize.py resumes per doc under --force. Needs candidates/ and
# clusters/ from M4 in the run dir (rsync them first) and the pass-1 audio under data/derived.
set -euo pipefail
cd "$SLURM_SUBMIT_DIR"
export PATH="$HOME/.local/bin:$PATH"
work_root="/work/hdd/bbjs/$USER"
export HF_HOME="$work_root/cache/huggingface"
export TORCH_HOME="$work_root/cache/torch"
export NLTK_DATA="$work_root/cache/nltk_data"
export UV_CACHE_DIR="$work_root/cache/uv"
export XDG_CACHE_HOME="$work_root/cache/xdg"

nvidia-smi --query-gpu=name,memory.total --format=csv
run=runs/e21_wlv3_pass1_espnet
test -d "$run/clusters" || { echo "$run/clusters missing: rsync the M4 candidates/ and clusters/ first"; exit 1; }

# the method: sum over occurrences, mean-token length norm, co-occurring mentions masked, n-best pool
uv run python scripts/canon/canonicalize.py --run-dir "$run" --canonicalizer b --model large-v3 --split all --expand-nbest --force
# masking ablation: same pool, other mentions of the cluster left in the window text
uv run python scripts/canon/canonicalize.py --run-dir "$run" --canonicalizer b --model large-v3 --split all --expand-nbest --no-mask-cooccurring --out-subdir lexicon_b_nomask --force

# CPU replays from the stored scores: length norms (M5.3), observed-only pool (M5.4), top-1 aggregation (E4)
for split in dev test; do
  uv run python scripts/canon/eval_lexicon.py --run-dir "$run" --lexicon-subdir lexicon_b --split "$split"
  uv run python scripts/canon/eval_lexicon.py --run-dir "$run" --lexicon-subdir lexicon_b --split "$split" --length-norm sum --tag sum
  uv run python scripts/canon/eval_lexicon.py --run-dir "$run" --lexicon-subdir lexicon_b --split "$split" --length-norm penalty:0.5 --tag penalty05
  uv run python scripts/canon/eval_lexicon.py --run-dir "$run" --lexicon-subdir lexicon_b --split "$split" --observed-only --tag observed
  uv run python scripts/canon/eval_lexicon.py --run-dir "$run" --lexicon-subdir lexicon_b --split "$split" --aggregate top1 --tag top1
  uv run python scripts/canon/eval_lexicon.py --run-dir "$run" --lexicon-subdir lexicon_b_nomask --split "$split"
done

# A0 over (b), then the transition matrix against pass 1
uv run python scripts/decode/pass2.py --run-dir "$run" --lexicon-subdir lexicon_b --variant a0 --out-run runs/e21_wlv3_a0_canonb --force
for split in dev test; do
  uv run python scripts/eval/evaluate.py --run-dir runs/e21_wlv3_a0_canonb --phase pass2 --split "$split" --compare-run "$run" --compare-phase pass1
done
