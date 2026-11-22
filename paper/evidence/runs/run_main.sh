#!/bin/bash
cd /root/attn-reuse
exec /opt/venv/bin/python reuse_real.py --seq 512 --n-batches 4 --ks 32,64,128,192,256,384 --groups 2,3,4,6,12 --out /root/attn-reuse/out/main.json
