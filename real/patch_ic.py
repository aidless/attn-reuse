"""Port the instrumentation hooks into the IndexCache modeling file.

The published `modeling_dots_dsa.py` is byte-identical across the DSA,
IndexCache and LatentIndex variants and carries no hooks, so the measurements in
`reports/26-trained-sharing-indexcache.md` require an instrumented copy. This
script is the instrument, kept as an artifact so the intervention is inspectable
rather than a local edit nobody can reproduce.

What it adds, and nothing else:

  SUBSET    layer -> {chunk: indices}   replacement sets, applied to followers
  CAPTURE   layer -> {chunk: indices}   recorded selections
  DO_CAPTURE[0]                          whether to record

The replacement happens AFTER the model computes its indices and BEFORE they are
turned into the boolean mask, so every downstream consumer (including the group
route handoff) sees the substituted set. The model's own arithmetic is untouched;
this is a probe, not a variant.

Usage:
    python3 patch_ic.py /root/models/DSV2-Lite-IndexCache-Sparse-1250
"""
import ast
import sys
import pathlib

MODULE = "modeling_dots_dsa.py"

GLOBALS = "SUBSET = {}\nCAPTURE = {}\nDO_CAPTURE = [False]\n\n\n"
ANCHOR_CLASS = "class DotsDSACache(DynamicCache):"

ANCHOR_TOPK = "        indices = scores.topk(min(c.index_topk, key_length), dim=-1).indices"
ANCHOR_MASK = "        selected = torch.zeros_like(valid).scatter_(-1, indices, True)"

INJECT = '''        # --- instrumentation (patch_ic.py) -------------------------------
        if DO_CAPTURE[0]:
            CAPTURE.setdefault(self.layer, {})[(start, end)] = indices.detach().clone()
        forced = SUBSET.get(self.layer)
        repl = forced.get((start, end)) if forced else None
        if repl is not None:
            indices = repl.to(indices.device)
        # --- end instrumentation -----------------------------------------


'''


def patch(path):
    src = path.read_text()
    if "DO_CAPTURE" in src:
        print("already instrumented")
        return False
    if src.count(ANCHOR_CLASS) != 1:
        raise SystemExit("class anchor not unique: %d" % src.count(ANCHOR_CLASS))
    if src.count(ANCHOR_TOPK) != 1:
        raise SystemExit("topk anchor not unique: %d" % src.count(ANCHOR_TOPK))
    if src.count(ANCHOR_MASK) != 1:
        raise SystemExit("mask anchor not unique: %d" % src.count(ANCHOR_MASK))
    src = src.replace(ANCHOR_CLASS, GLOBALS + ANCHOR_CLASS)
    src = src.replace(ANCHOR_TOPK, ANCHOR_TOPK + "\n" + INJECT.rstrip("\n"))
    path.write_text(src)
    ast.parse(src)                      # refuse to leave an unparseable file
    return True


def main():
    root = pathlib.Path(sys.argv[1] if len(sys.argv) > 1 else
                        "/root/models/DSV2-Lite-IndexCache-Sparse-1250")
    target = root / MODULE
    if not target.exists():
        raise SystemExit("not found: %s" % target)
    changed = patch(target)
    src = target.read_text()
    print("changed:", changed, "| markers:",
          sum(src.count(k) for k in ("SUBSET", "CAPTURE", "DO_CAPTURE")))


if __name__ == "__main__":
    main()
