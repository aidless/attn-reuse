#!/bin/bash
# Probe stage only: measure dense MMLU accuracy under the two scoring metrics
# before spending GPU hours on a grid whose subject may be at chance.
#
# The first MMLU attempt reported full acc = 0.2300 (chance = 0.25) on seq 384
# with 5 shots, because the prompt was truncated down to the demonstrations and
# the question itself was cut off. After fixing the truncation the same 40-item
# probe still gave 0.1750 (0.5B) and 0.2000 (1.5B) at seq 640 — which means the
# truncation was not the only problem: sum-of-logprob over options of very
# different length is biased toward short answers, and on MMLU (unlike ARC) the
# option lengths range from 2 to 20 tokens.
#
# So: measure both metrics on the same 60 items. If length-normalised scoring
# puts a model meaningfully above chance, the grid runs with --length-norm.
# If neither does, MMLU is not a usable second bench for these small models and
# the honest move is to say so rather than to report a 6-point swing on a
# nearly-random classifier.
set -u
cd /root/attn-reuse || exit 2
PY=/opt/venv/bin/python
export HF_ENDPOINT=https://hf-mirror.com
mkdir -p logs

for m in Qwen2.5-0.5B Qwen2.5-1.5B Qwen2.5-7B; do
  echo "--- probe $m $(date -u)"
  for norm in raw lengthnorm; do
    extra=""
    [ "$norm" = "lengthnorm" ] && extra="--length-norm"
    $PY mcq_reuse.py --model "models/$m" --bench mmlu --shots 5 --n 60 --seq 640 \
        --ks 256 --groups 3 --dtype bfloat16 $extra --out "/tmp/probe_${m}_${norm}.json" \
        2>&1 | grep -E "shots|items=|full acc|control ok" | sed "s/^/  [$norm] /"
  done
  echo "--- $m done $(date -u)"
done
echo "=== PROBE DONE $(date -u) ==="
