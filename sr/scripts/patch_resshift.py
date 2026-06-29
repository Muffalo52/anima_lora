#!/usr/bin/env python
"""Idempotently patch the ResShift clone for the no-xformers / Blackwell env.

ResShift/ is a gitignored upstream clone, so these source edits don't live in our
git history — this script re-applies them after a fresh `git clone`. Two patches:

1. VQGAN vanilla AttnBlock (ldm/modules/diffusionmodules/model.py): the single-head
   head_dim=512 attention over ~65k tokens has no flash/efficient SDPA kernel and
   the math backend materializes a 16 GiB map → OOM. Swap to a query-chunked exact
   SDPA (bit-faithful, O(Bq*N) memory). xformers is intentionally absent.

Run:  python sr/scripts/patch_resshift.py
"""
from pathlib import Path

RESSHIFT = Path(__file__).resolve().parents[2] / "ResShift"
MODEL = RESSHIFT / "ldm" / "modules" / "diffusionmodules" / "model.py"

HELPER = '''
# Anima SR sidecar: memory-efficient attention without xformers (Blackwell).
# The VQGAN mid-attention is SINGLE-HEAD with head_dim=512 over a ~65k-token seq.
# That exceeds flash/efficient SDPA kernel limits (head_dim>256) so SDPA would use
# the O(N^2) math backend and OOM (16 GiB map). Splitting heads would change the
# (trained) single-head math. Instead we chunk over the query axis: each block does
# exact attention against all keys, so memory is O(Bq*N) not O(N*N) — bit-faithful.
def _sdpa_efficient(q, k, v, q_chunk=2048):  # q,k,v: (b, L, c)
    L = q.shape[1]
    if L <= q_chunk:
        return torch.nn.functional.scaled_dot_product_attention(q, k, v)
    outs = []
    for i in range(0, L, q_chunk):
        outs.append(
            torch.nn.functional.scaled_dot_product_attention(q[:, i:i + q_chunk], k, v)
        )
    return torch.cat(outs, dim=1)
'''

ANCHOR = '    print("No module \'xformers\'. Proceeding without it.")\n'

OLD_ATTN = '''        # compute attention
        b,c,h,w = q.shape
        q = q.reshape(b,c,h*w)
        q = q.permute(0,2,1)   # b,hw,c
        k = k.reshape(b,c,h*w) # b,c,hw
        w_ = torch.bmm(q,k)     # b,hw,hw    w[b,i,j]=sum_c q[b,i,c]k[b,c,j]
        w_ = w_ * (int(c)**(-0.5))
        w_ = torch.nn.functional.softmax(w_, dim=2)

        # attend to values
        v = v.reshape(b,c,h*w)
        w_ = w_.permute(0,2,1)   # b,hw,hw (first hw of k, second of q)
        h_ = torch.bmm(v,w_)     # b, c,hw (hw of q) h_[b,c,j] = sum_i v[b,c,i] w_[b,i,j]
        h_ = h_.reshape(b,c,h,w)'''

NEW_ATTN = '''        # compute attention via query-chunked exact SDPA (Anima SR sidecar,
        # Blackwell / no-xformers) so the b,hw,hw map is never materialized.
        b,c,h,w = q.shape
        q = q.reshape(b,c,h*w).permute(0,2,1)   # b,hw,c
        k = k.reshape(b,c,h*w).permute(0,2,1)   # b,hw,c
        v = v.reshape(b,c,h*w).permute(0,2,1)   # b,hw,c
        h_ = _sdpa_efficient(q, k, v)  # b,hw,c
        h_ = h_.permute(0,2,1).reshape(b,c,h,w)'''


def main():
    if not MODEL.exists():
        raise SystemExit(f"ResShift model.py not found at {MODEL} — clone ResShift first.")
    src = MODEL.read_text()
    if "_sdpa_efficient" in src:
        print("model.py already patched — skipping.")
        return
    if ANCHOR not in src or OLD_ATTN not in src:
        raise SystemExit(
            "Upstream model.py differs from the expected text — patch by hand "
            "(see sr/README.md 'ResShift source patches')."
        )
    src = src.replace(ANCHOR, ANCHOR + HELPER, 1)
    src = src.replace(OLD_ATTN, NEW_ATTN, 1)
    MODEL.write_text(src)
    print(f"patched {MODEL}")


if __name__ == "__main__":
    main()
