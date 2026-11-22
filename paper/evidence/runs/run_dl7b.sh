set -x
export HF_ENDPOINT=https://hf-mirror.com
/opt/venv/bin/python - <<'PY'
from huggingface_hub import snapshot_download
snapshot_download("Qwen/Qwen2.5-7B", local_dir="/root/attn-reuse/models/Qwen2.5-7B",
                  allow_patterns=["*.json","*.safetensors","*.txt"])
print("DONE")
PY
du -sh /root/attn-reuse/models/Qwen2.5-7B
