#!/bin/bash
cd /root/attn-reuse
M=/root/attn-reuse/models/Qwen2.5-1.5B
/opt/venv/bin/python reuse_real.py --seq 512 --n-batches 4 --ks 128,256,384 --groups 2,3,6 --dtype bfloat16 --model $M --out /root/attn-reuse/out/main_15b.json
/opt/venv/bin/python sweep2.py --mode pairs --k 256 --window 6 --dtype bfloat16 --model $M --out /root/attn-reuse/out/pairs_15b.jsonl
