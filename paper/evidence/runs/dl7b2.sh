export HF_ENDPOINT=https://hf-mirror.com
M=/root/attn-reuse/models/Qwen2.5-7B
mkdir -p "$M"
BASE=https://hf-mirror.com/Qwen/Qwen2.5-7B/resolve/main
for f in config.json generation_config.json tokenizer.json tokenizer_config.json vocab.json merges.txt model.safetensors.index.json; do
  [ -s "$M/$f" ] && { echo "skip $f"; continue; }
  echo "get $f"
  curl -sSL --retry 3 -o "$M/$f" "$BASE/$f" || echo "FAIL $f"
done
for i in 1 2 3 4; do
  f=model-0000${i}-of-00004.safetensors
  [ -s "$M/$f" ] && { echo "skip $f"; continue; }
  echo "get $f"
  curl -sSL --retry 3 -o "$M/$f.part" "$BASE/$f" && mv "$M/$f.part" "$M/$f" || echo "FAIL $f"
done
echo DOWNLOAD_DONE
du -sh "$M"
