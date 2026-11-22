#!/bin/bash
cd /root/attn-reuse
export HF_HOME=/root/attn-reuse/models HF_ENDPOINT=https://hf-mirror.com
/opt/venv/bin/python mcq_reuse.py --model models/Qwen2.5-0.5B --bench arc_challenge \
  --n 60 --shots 8 --seq 512 --ks 64 --groups 3 \
  --out out/probe_arcc_05b.json > logs/probe_arcc_05b.log 2>&1
echo "EXIT=$?"
