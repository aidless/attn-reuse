#!/bin/bash
# Second benchmark for the decision layer: MMLU high_school_mathematics.
#
# Why a second bench: every decision-level number in report 07 comes from
# ARC-Easy. The directionality result (b:c ~5:1 on 1.5B/7B) is the most
# publishable single point, and a claim about a direction of error needs a
# second substrate before it is a finding rather than a property of one
# question set. ARC-Easy is largely retrieval-style science questions; MMLU
# high_school_mathematics is short arithmetic reasoning, a different failure
# mode for an attention perturbation.
#
# FIRST ATTEMPT FAILED, and the failure is instructive: with seq 384 and 5
# shots, `prompt_len median=380` meant the prompt had been trimmed to the
# budget, and what survived was the demonstrations — the question being asked
# was cut off. Measured "full acc=0.2300" against chance 0.25. That is a
# plausible-looking number produced by a prompt that never states the problem.
# The instrument now keeps the question and drops shots instead; seq goes to
# 640 so that at least a couple of shots still fit.
set -u
cd /root/attn-reuse || exit 2
PY=/opt/venv/bin/python
export HF_ENDPOINT=https://hf-mirror.com
mkdir -p logs

run () {  # $1 tag  $2 model  $3 n  $4 ks  $5 groups
  if [ -f "out/$1.json" ]; then echo "SKIP $1 (exists)"; return 0; fi
  echo "=== $1 start $(date -u) ==="
  $PY mcq_reuse.py --model "models/$2" --bench mmlu --shots 5 --n "$3" --seq 640 \
      --ks "$4" --groups "$5" --dtype bfloat16 --out "out/$1.json" \
      || echo "FAILED $1 rc=$?"
  echo "=== $1 done $(date -u) ==="
}

run mmlu_05b Qwen2.5-0.5B  200 64,128,256,384 2,3,4,6
run mmlu_15b Qwen2.5-1.5B  200 64,128,256     2,3,6
run mmlu_7b  Qwen2.5-7B    120 128,256        2,3
echo "=== ALL DONE $(date -u) ==="
