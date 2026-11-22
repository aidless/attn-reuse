#!/bin/bash
cd /root/attn-reuse
/opt/venv/bin/python /root/attn-reuse/margin_probe.py --seq 512 --k 256 --n-batches 4 --dtype float16 --out /root/attn-reuse/out/margin_512.json
