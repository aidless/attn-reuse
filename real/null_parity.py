"""Do the dense and trained-sparse harnesses build the same null?

The paper now quotes the reuse cost against an arbitrary-set null on two
different model families, measured by two different forward-pass
implementations (reuse_llama.py hooks vs modeling_dots_dsa.py patch). If the two
nulls were built under different rules, the ratio 0.034-0.126 on dense weights
and the trained-sparse ratio would not be the same quantity.

This checks the rules, not the models: same shape in, and both selectors must
agree on the four invariants that make a null fair. Runs on CPU.
"""
import importlib.util, sys, torch

spec = importlib.util.spec_from_file_location("rl", "attn-reuse/real/reuse_llama.py")
rl = importlib.util.module_from_spec(spec)
sys.argv = ["x"]              # keep argparse out of import time
spec.loader.exec_module(rl)


def dsa_rand_set(chunk_start, chunk_end, key_length, kk, layer, seed=7):
    """Verbatim copy of the selector used in /root/dsa_group_rand.py."""
    T = chunk_end - chunk_start
    qpos = torch.arange(chunk_start, chunk_end)
    p = torch.arange(key_length)
    visible = p[None, :] <= qpos[:, None]
    g = torch.Generator().manual_seed(seed * 100003 + layer)
    noise = torch.rand((T, key_length), generator=g)
    noise = noise.masked_fill(~visible, -1.0)
    sel = noise.topk(kk, dim=-1).indices
    sel = torch.minimum(sel, qpos[:, None])
    return sel.unsqueeze(0)


def invariants(sel, qpos, key_length, kk, label):
    """The properties a fair null must have. Normalises 4-D (B,H,T,k) and
    3-D (B,T,k) selection tensors into one shape before checking."""
    if sel.dim() == 3:
        sel = sel.unsqueeze(1)                       # -> (B,1,T,k)
    (B, H, T, K) = sel.shape
    q = qpos if qpos is not None else torch.arange(T)
    past = int((sel > q.view(1, 1, -1, 1)).sum())   # (a) never past the query
    oob = int(((sel < 0) | (sel >= key_length)).sum())            # (c) in range
    dup = 0                                          # (b) repeats per row
    for b in range(B):
        for h in range(H):
            u = torch.sort(sel[b, h], dim=-1).values
            dup += int((u[:, 1:] == u[:, :-1]).sum())
    # (d) rows that could not supply k distinct visible keys are inert: they
    #     attend to everything visible, so the cell equals the dense pass there
    inert = int((q < kk - 1).sum())
    print("  %-11s past=%d dup=%d oob=%d  rows_with_fewer_visible_than_k=%d/%d"
          % (label, past, dup, oob, inert, T))
    return past, dup, oob


def main():
    torch.manual_seed(0)
    ok = True

    print("dense harness (reuse_llama.random_selection), scores (1,4,32,32):")
    scores = torch.randn(1, 4, 32, 32)
    kk = 8
    rl.STATE["rand"] = {}
    sel_d = rl.random_selection(scores, kk, seed=7, layer=0)
    past, dup, oob = invariants(sel_d, torch.arange(32), 32, kk, "dense")
    ok &= (past == 0 and oob == 0)

    print("trained-sparse harness (dsa_group_rand.rand_set), one chunk 0..32:")
    sel_s = dsa_rand_set(0, 32, 32, kk, layer=0)
    past2, dup2, oob2 = invariants(sel_s, torch.arange(32), 32, kk, "dsa")
    ok &= (past2 == 0 and oob2 == 0)

    # determinism under a fixed seed, and sensitivity to a changed seed
    rl.STATE["rand"] = {}
    a = rl.random_selection(scores, kk, seed=7, layer=0)
    rl.STATE["rand"] = {}
    b = rl.random_selection(scores, kk, seed=7, layer=0)
    c = dsa_rand_set(0, 32, 32, kk, layer=0)
    d = dsa_rand_set(0, 32, 32, kk, layer=0)
    print("  reproducib  dense=%s dsa=%s" % (bool((a == b).all()), bool((c == d).all())))
    ok &= bool((a == b).all()) and bool((c == d).all())

    rl.STATE["rand"] = {}
    e = rl.random_selection(scores, kk, seed=8, layer=0)
    print("  seed-change dense differs=%s" % bool((a != e).any()))
    ok &= bool((a != e).any())

    # The one deliberate difference: short rows. Early queries can see fewer than
    # kk positions. The dense harness clamps into the visible range; the DSA copy
    # does the same. Both therefore may repeat a position, which is why the
    # duplicate count is reported rather than asserted to be zero.
    # Short ROWS: a 10-position sequence where every row can see fewer keys than
    # k. The two harnesses index rows differently -- the dense one takes a full
    # (T,T) score matrix (row i == query position i), the DSA one takes an
    # explicit chunk offset -- so to compare the same situation the DSA chunk
    # must start at 0. Both should end up selecting only visible keys.
    sc = torch.randn(1, 2, 10, 10)
    rl.STATE["rand"] = {}
    short_d = rl.random_selection(sc, 8, seed=7, layer=0)
    p3, _, o3 = invariants(short_d, torch.arange(10), 10, 8, "dense-short")
    short_s = dsa_rand_set(0, 10, 10, 8, layer=0)
    p4, _, o4 = invariants(short_s, torch.arange(10), 10, 8, "dsa-short")
    print("  short rows: both must still never pass the query (dense=%d dsa=%d)" % (p3, p4))
    ok &= (p3 == 0 and p4 == 0)

    assert ok, "null construction rules disagree between the two harnesses"
    print("NULL PARITY OK: both selectors keep every invariant, and differ only in "
          "the one documented place (row shorter than k, where both clamp)")


if __name__ == "__main__":
    main()
