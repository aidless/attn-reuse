#!/bin/bash
# Second substrate for the directionality claim: ARC-Challenge, 1.5B and 7B only.
#
# Dense accuracy measured on 40 items before committing GPU hours:
#   0.5B  0.2750  (chance = 0.2500)  -> excluded, no headroom
#   1.5B  0.4750                     -> headroom
#   7B    0.6000                     -> headroom
# The 0.5B is left out on purpose. A model solving 27.5% of a four-way test is
# answering at chance; a perturbation measured there reports the model's floor,
# not the reuse's cost. ARC-Easy already covers 0.5B at 0.6100.
#
# What this run is for: report 07's strongest single point is that the flipped
# answers are overwhelmingly correct->wrong (1.5B b:c = 5.13:1, 7B 5.0:1).
# A ratio about the *direction* of error needs a second question distribution
# before it is a finding. ARC-Challenge answers need a step of reasoning where
# ARC-Easy mostly needs recall, and its dense accuracy is 0.475/0.600 instead
# of 0.760/0.793 — i.e. more room for damage to show.
set -u
cd /root/attn-reuse || exit 2
PY=/opt/venv/bin/python
export HF_ENDPOINT=https://hf-mirror.com
mkdir -p logs

run () {  # $1 tag  $2 model  $3 n  $4 ks  $5 groups
  if [ -f "out/$1.json" ]; then echo "SKIP $1 (exists)"; return 0; fi
  echo "=== $1 start $(date -u) ==="
  $PY mcq_reuse.py --model "models/$2" --bench arc_challenge --shots 8 --n "$3" \
      --seq 384 --ks "$4" --groups "$5" --dtype bfloat16 --out "out/$1.json" \
      || echo "FAILED $1 rc=$?"
  echo "=== $1 done $(date -u) ==="
}

run arcc_15b Qwen2.5-1.5B  200 64,128,256 2,3,6
run arcc_7b  Qwen2.5-7B    120 128,256    2,3
echo "=== ALL DONE $(date -u) ==="
