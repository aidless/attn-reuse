#!/bin/bash
# 7B ARC-Challenge at k=64, so the 7B grid matches the 1.5B grid in k coverage.
# Reason: at k=128 the 7B already shows -9.2 pt (p=0.0074) while k=256 is flat.
# Without k=64 the report can only say "the smallest k we tried on 7B was 128",
# which leaves open that the damage is much worse below it — the shape claim
# (cost falls with k) needs at least three k values on every model.
set -u
cd /root/attn-reuse || exit 2
PY=/opt/venv/bin/python
export HF_ENDPOINT=https://hf-mirror.com
mkdir -p logs
if [ -f out/arcc_7b_k64.json ]; then echo "SKIP (exists)"; exit 0; fi
echo "=== arcc_7b_k64 start $(date -u) ==="
$PY mcq_reuse.py --model models/Qwen2.5-7B --bench arc_challenge --shots 8 --n 120 \
    --seq 384 --ks 64 --groups 2,3 --dtype bfloat16 --out out/arcc_7b_k64.json \
    || echo "FAILED rc=$?"
echo "=== done $(date -u) ==="
