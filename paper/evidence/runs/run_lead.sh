#!/bin/bash
cd /root/attn-reuse
/opt/venv/bin/python /root/attn-reuse/leader_anomaly.py --seq 512 --k 256 --n-batches 4 --dtype float16 --out /root/attn-reuse/out/leader_512.json
