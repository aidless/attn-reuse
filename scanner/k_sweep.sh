#!/usr/bin/env bash
set -u
cd "$(dirname "$0")/.."
for k in 4 8 16 32; do
  for spec in "openai-community/gpt2:gpt2" "distilbert/distilgpt2:distilgpt2"; do
    model="${spec%%:*}"; short="${spec##*:}"
    out="data/${short}_k${k}.json"
    [ -s "$out" ] && { echo "skip $out"; continue; }
    echo "=== $short k=$k ==="
    HF_ENDPOINT=https://hf-mirror.com timeout 900 python3 scanner/layer_reuse.py \
      --model "$model" --k "$k" --json "$out" > "data/${short}_k${k}.log" 2>&1
    echo "rc=$? -> $out"
  done
done
echo SWEEP_DONE
