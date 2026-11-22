"""Instrument the LatentIndex checkpoint's modeling file.

The published `modeling_dots_dsa.py` is byte-identical across the DSA,
IndexCache and LatentIndex families and carries no hooks, so every
intervention this project performs has to be re-applied to a fresh
download.  This script does that idempotently and refuses to leave a
file behind that does not parse.

Two hooks are installed:

* ``SUBSET`` / ``CAPTURE`` / ``DO_CAPTURE`` -- the same index-substitution
  hook used for the IndexCache and DSA checkpoints.  ``SUBSET[layer]`` is a
  dict keyed by ``(chunk_start, chunk_end)`` whose value replaces the
  selected indices for that chunk.
* ``ROUTE_OFFSET`` -- new here.  A LatentIndex follower reads its own
  offset out of the leader's route list (``group[self.offset]``), so it
  already scores with its own projection.  ``ROUTE_OFFSET[layer] = k``
  forces that layer to read ``group[k]`` instead, which is how an
  IndexCache follower behaves (every follower reads offset 0).  Forcing a
  follower to offset 0 therefore measures exactly what the layer-specific
  scoring buys.
"""
import ast
import sys

GLOBALS = '''SUBSET = {}
CAPTURE = {}
DO_CAPTURE = [False]
ROUTE_OFFSET = {}


'''

ANCHOR_CLASS = "class DotsDSACache(DynamicCache):"

ANCHOR_ROUTE = ("            route = group[self.offset if c.dots_dsa_method == "
                "'latentindexer' and self.group_size > 1 else 0]\n")

ROUTE_PATCHED = '''            _off = ROUTE_OFFSET.get(self.layer, None)
            route = group[_off if _off is not None else (self.offset if c.dots_dsa_method == 'latentindexer' and self.group_size > 1 else 0)]
'''

ANCHOR_TOPK = "                indices = scores.topk(min(c.index_topk, key_length), dim=-1).indices\n"

TOPK_PATCHED = '''                indices = scores.topk(min(c.index_topk, key_length), dim=-1).indices
                if DO_CAPTURE[0]:
                    CAPTURE.setdefault(self.layer, {})[(start, end)] = indices.detach().clone()
                _forced = SUBSET.get(self.layer)
                if _forced is not None:
                    _repl = _forced.get((start, end))
                    if _repl is not None:
                        indices = _repl.to(indices.device)
'''

ANCHOR_MASK = "                selected = torch.zeros_like(valid).scatter_(-1, indices, True)\n"


def patch(path):
    src = open(path, encoding="utf-8").read()
    if "ROUTE_OFFSET" in src:
        return "already patched"
    for name, anchor in (("class", ANCHOR_CLASS), ("route", ANCHOR_ROUTE),
                         ("topk", ANCHOR_TOPK), ("mask", ANCHOR_MASK)):
        if src.count(anchor) != 1:
            raise SystemExit(f"anchor {name} appears {src.count(anchor)} times, expected 1")
    src = src.replace(ANCHOR_CLASS, GLOBALS + ANCHOR_CLASS, 1)
    src = src.replace(ANCHOR_ROUTE, ROUTE_PATCHED, 1)
    src = src.replace(ANCHOR_TOPK, TOPK_PATCHED, 1)
    ast.parse(src)
    open(path, "w", encoding="utf-8").write(src)
    return "patched"


if __name__ == "__main__":
    print(patch(sys.argv[1]))
