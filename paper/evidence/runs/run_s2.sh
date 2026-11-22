#!/bin/bash
cd /root/attn-reuse
/opt/venv/bin/python sweep2.py --mode pairs --k 256 --window 4 --out /root/attn-reuse/out/pairs.jsonl
/opt/venv/bin/python sweep2.py --mode offset --k 256 --offset 1 --out /root/attn-reuse/out/offset.jsonl
