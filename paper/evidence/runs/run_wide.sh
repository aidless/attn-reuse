#!/bin/bash
cd /root/attn-reuse
/opt/venv/bin/python sweep2.py --mode pairs --k 256 --window 12 --out /root/attn-reuse/out/pairs_wide.jsonl
