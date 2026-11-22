export HF_ENDPOINT=https://hf-mirror.com
cd /root/attn-reuse
P=/opt/venv/bin/python
echo "=== 0.5B ==="; date -u
$P mcq_reuse.py --model models/Qwen2.5-0.5B --bench arc_easy --shots 8 \
   --n 200 --seq 384 --ks 64,128,256 --groups 2,3,6 --out out/mcq_05b.json
echo "=== 1.5B ==="; date -u
$P mcq_reuse.py --model models/Qwen2.5-1.5B --bench arc_easy --shots 8 \
   --n 200 --seq 384 --ks 64,128,256 --groups 2,3,6 --out out/mcq_15b.json
echo "=== 7B ==="; date -u
$P mcq_reuse.py --model models/Qwen2.5-7B --bench arc_easy --shots 8 \
   --n 150 --seq 384 --ks 128,256 --groups 2,3,6 --out out/mcq_7b.json
echo "=== ALL DONE ==="; date -u
