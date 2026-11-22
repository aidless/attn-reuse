export HF_ENDPOINT=https://hf-mirror.com
cd /root/attn-reuse
P=/opt/venv/bin/python
echo "=== offset 0.5B k64 g6 ==="; date -u
$P mcq_offset.py --model models/Qwen2.5-0.5B --n 120 --shots 8 --seq 384 \
   --k 64 --group 6 --out out/mcq_offset_05b_k64g6.json
echo "=== offset 0.5B k128 g6 ==="; date -u
$P mcq_offset.py --model models/Qwen2.5-0.5B --n 120 --shots 8 --seq 384 \
   --k 128 --group 6 --out out/mcq_offset_05b_k128g6.json
echo "=== OFFSET DONE ==="; date -u
