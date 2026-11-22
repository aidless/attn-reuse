#!/bin/bash
# Second substrate for the decision layer, on ARC-Challenge.
#
# Why not MMLU: the probe measured dense accuracy 0.1500 / 0.1667 on 0.5B
# (chance 0.25) and 0.2000 on 1.5B, under both the raw and the length-normalised
# scoring metric. A classifier below chance has no headroom to lose, so a
# "6-point drop" measured there is a story about the model, not about the
# reuse. Reporting it as a second benchmark would be reporting noise as
# replication.
#
# ARC-Challenge is the same instrument and the same format, on a split where
# 1.5B sits around 0.6 instead of 0.76 — enough headroom for a perturbation to
# be visible, and a genuinely different question distribution (multi-step
# reasoning rather than recall) from ARC-Easy. The point of the second run is
# the *direction* of the error (b:c), which only needs a non-saturated cell.
set -u
cd /root/attn-reuse || exit 2
PY=/opt/venv/bin/python
export HF_ENDPOINT=https://hf-mirror.com
mkdir -p logs

run () {  # $1 tag  $2 model  $3 n  $4 ks  $5 groups
  if [ -f "out/$1.json" ]; then echo "SKIP $1 (exists)"; return 0; fi
  echo "=== $1 start $(date -u) ==="
  $PY mcq_reuse.py --model "models/$2" --bench arc_challenge --shots 8 --n "$3" \
      --seq 384 --ks "$4" --groups "$5" --dtype bfloat16 --out "out/$1.json" \
      || echo "FAILED $1 rc=$?"
  echo "=== $1 done $(date -u) ==="
}

run arcc_05b Qwen2.5-0.5B  200 64,128,256   2,3,4,6
run arcc_15b Qwen2.5-1.5B  200 64,128,256   2,3,6
run arcc_7b  Qwen2.5-7B    120 128,256       2,3
echo "=== ALL DONE $(date -u) ==="
