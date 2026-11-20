"""The attention fallback used off CUDA (Mac MPS, CPU, fp32) must keep a finite mask: MPS turned the -inf
entries (key padding to a multiple of 16, the negative rows that skip the reference) into black images.
Results must not change: compared with the pre-fix code and a float64 softmax written out by hand."""
import math
import unittest
from unittest.mock import patch

import torch
import torch.nn.functional as F

from snapref_forge import core


def reference_masked_attention(q, k_t, v_t, kv_ref, sizes, biases, rows=None, row_biases=None):
    """_masked_attention before the fix (padding and -inf on every device)."""
    k_r, v_r = kv_ref
    B, S, H, D = q.shape
    k = torch.cat([k_r.expand(B, *k_r.shape[1:]).to(q.dtype), k_t], dim=1).transpose(1, 2)
    v = torch.cat([v_r.expand(B, *v_r.shape[1:]).to(q.dtype), v_t], dim=1).transpose(1, 2)
    n = k.shape[2]
    pad = (-n) % 16
    if pad:
        k, v = F.pad(k, (0, 0, 0, pad)), F.pad(v, (0, 0, 0, pad))
    mask = torch.zeros(B if rows is not None else 1, 1, 1, n + pad, device=q.device, dtype=q.dtype)
    start = 0
    for size, bias in zip(sizes, biases):
        mask[..., start:start + size] = bias
        start += size
    if rows is not None:
        selected = torch.zeros(B, device=q.device, dtype=torch.bool)
        selected[rows] = True
        mask[~selected, ..., :start] = float("-inf")
        if row_biases is not None:
            mask[rows, ..., :start] += torch.as_tensor(row_biases, device=q.device, dtype=q.dtype).view(-1, 1, 1, 1)
    if pad:
        mask[..., n:] = float("-inf")
    out = F.scaled_dot_product_attention(q.transpose(1, 2), k, v, attn_mask=mask.expand(B, H, S, n + pad))
    return out.transpose(1, 2).reshape(B, S, H * D)


def exact(q, k_t, v_t, kv_ref, sizes, biases, rows, row_biases):
    q, k_t, v_t = q.double(), k_t.double(), v_t.double()
    k_r, v_r = (t.double()[0] for t in kv_ref)
    B, S, H, D = q.shape
    ref_bias = torch.cat([torch.full((s,), float(b), dtype=torch.float64) for s, b in zip(sizes, biases)])
    out = []
    for b in range(B):
        if b in rows:
            extra = 0.0 if row_biases is None else row_biases[list(rows).index(b)]
            keys, vals = torch.cat([k_r, k_t[b]]), torch.cat([v_r, v_t[b]])
            bias = torch.cat([ref_bias + extra, torch.zeros(k_t.shape[1], dtype=torch.float64)])
        else:
            keys, vals, bias = k_t[b], v_t[b], torch.zeros(k_t.shape[1], dtype=torch.float64)
        logits = torch.einsum("shd,khd->hsk", q[b], keys) / math.sqrt(D) + bias
        out.append(torch.einsum("hsk,khd->shd", logits.softmax(-1), vals).reshape(S, H * D))
    return torch.stack(out)


def inputs(B, S, sizes, dtype=torch.float32, H=2, D=16):
    g = torch.Generator().manual_seed(1)
    r = lambda *s: torch.randn(*s, generator=g).to(dtype)   # noqa: E731
    return r(B, S, H, D), r(B, S, H, D), r(B, S, H, D), (r(1, sum(sizes), H, D), r(1, sum(sizes), H, D))


# Forge always passes rows: positive rows only (default), every row (negative branch / Fidelity)
CASES = [
    ("positive only, one ref, n % 16 == 0", 2, 48, [32], [0.0], [0], [0.0]),
    ("positive only, one ref, n % 16 != 0", 2, 45, [36], [0.0], [0], [0.0]),
    ("positive only, refs weighted 1 / 0.4, n % 16 != 0", 2, 45, [20, 13], [0.0, math.log(0.4)], [0], [0.0]),
    ("strength 1.5 on two images per branch, n % 16 != 0", 4, 45, [30], [math.log(1.5)], [0, 1], [0.0, 0.0]),
    ("Fidelity 0.5, n % 16 != 0", 2, 45, [36], [0.0], [0, 1], [0.0, math.log(0.5)]),
    ("negative branch reads fully, refs weighted, n % 16 == 0", 2, 46, [20, 14], [0.0, math.log(1.6)], [0, 1], [0.0, 0.0]),
]


class MaskedAttentionTests(unittest.TestCase):
    def test_unchanged_results(self):
        for dtype, tol in ((torch.float32, 1e-5), (torch.bfloat16, 3e-2), (torch.float16, 3e-2)):
            for name, B, S, sizes, biases, rows, row_biases in CASES:
                q, k, v, ref = inputs(B, S, sizes, dtype)
                with self.subTest(name, dtype=dtype):
                    out = core.attention(q, k, v, ref, sizes, biases, rows, row_biases)
                    self.assertEqual(out.dtype, dtype)
                    old = reference_masked_attention(q, k, v, ref, sizes, biases, rows, row_biases)
                    torch.testing.assert_close(out.float(), old.float(), rtol=tol, atol=tol)
                    torch.testing.assert_close(out.double(), exact(q, k, v, ref, sizes, biases, rows, row_biases),
                                               rtol=tol, atol=tol)

    def test_finite_masks_off_cuda(self):
        real, masks = F.scaled_dot_product_attention, []

        def strict_sdpa(q, k, v, attn_mask=None, **kw):   # stands in for MPS
            masks.append(attn_mask)
            out = real(q, k, v, attn_mask=attn_mask, **kw)
            return out if attn_mask is None or torch.isfinite(attn_mask).all() else torch.full_like(out, float("nan"))

        with patch.object(core.F, "scaled_dot_product_attention", strict_sdpa):
            for name, B, S, sizes, biases, rows, row_biases in CASES:
                q, k, v, ref = inputs(B, S, sizes)
                with self.subTest(name):
                    self.assertTrue(torch.isfinite(core.attention(q, k, v, ref, sizes, biases, rows, row_biases)).all())
        self.assertTrue(all(m is None or torch.isfinite(m).all() for m in masks))


if __name__ == "__main__":
    unittest.main()
