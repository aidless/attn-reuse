#!/bin/bash
cd /root/attn-reuse
/opt/venv/bin/python reuse_real.py --seq 2048 --n-batches 2 --ks 128,256,512 --groups 2,3,6 --out /root/attn-reuse/out/long.json
/opt/venv/bin/python precision.py --seq 2048 --n-batches 10 --out /root/attn-reuse/out/long_prec.json
