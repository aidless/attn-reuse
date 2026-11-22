#!/bin/bash
# ARC-Easy at 5 shots, to shot-match the ARC-Challenge runs.
# Why: mcq_reuse.py prepends "the longest prefix of exemplars that still fits",
# so ARC-Challenge (longer stems) actually used 5 exemplars while ARC-Easy used
# 8. Comparing damage/rescue across the two benchmarks would then mix a change
# of question distribution with a change of shot count. This run removes the
# second change: same 5 exemplars on ARC-Easy, same 0.5B model, same k/g cells
# as the ARC-Challenge 1.5B grid's lower half.
set -u
cd /root/attn-reuse || exit 2
PY=/opt/venv/bin/python
export HF_ENDPOINT=https://hf-mirror.com
mkdir -p logs
if [ -f out/arc_easy_5shot.json ]; then echo "SKIP (exists)"; exit 0; fi
echo "=== arc_easy 5-shot start $(date -u) ==="
$PY mcq_reuse.py --model models/Qwen2.5-0.5B --bench arc_easy --shots 5 --n 200 \
    --seq 384 --ks 64,128 --groups 2,3,6 --dtype bfloat16 \
    --out out/arc_easy_5shot.json || echo "FAILED rc=$?"
echo "=== done $(date -u) ==="
