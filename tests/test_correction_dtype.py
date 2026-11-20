"""Extra reference guidance with APG on backends without float64 (Mac MPS, Intel XPU): the correction falls back
to float32 there, and keeps exactly the old float64 result where float64 works (CPU, CUDA)."""
import os
import unittest
from unittest.mock import patch

import torch
import torch.nn.functional as F

from snapref_forge import core


def reference_apg(guided, pos):
    """APG as written before the fallback (always float64)."""
    dims = tuple(range(1, pos.ndim))
    update = (guided - pos).double()
    unit = F.normalize(pos.double().flatten(1), dim=1).view_as(pos)
    return pos + (update - (update * unit).sum(dim=dims, keepdim=True) * unit).to(pos.dtype)


class CorrectionDtypeTests(unittest.TestCase):
    def setUp(self):
        self.addCleanup(core._FP64_ARITH.update, dict(core._FP64_ARITH))
        core._FP64_ARITH.clear()
        env = patch.dict(os.environ, {}, clear=False)
        env.start()
        self.addCleanup(env.stop)
        os.environ.pop("SNAPREF_FP64", None)

    def test_float64_backends_keep_the_old_result(self):
        torch.manual_seed(0)
        guided, pos = torch.randn(2, 16, 1, 32, 48), torch.randn(2, 16, 1, 32, 48)
        self.assertTrue(torch.equal(core._correct(guided, pos, "apg"), reference_apg(guided, pos)))

    def test_float32_fallback_is_close_and_keeps_dtype(self):
        torch.manual_seed(0)
        pos = torch.randn(1, 16, 1, 128, 128)
        guided = pos + 4.5 * torch.randn_like(pos)
        with patch.object(core, "_fp64_arith_ok", lambda device: False):
            out = core._correct(guided, pos, "apg")
        self.assertEqual(out.dtype, pos.dtype)
        self.assertLess((out.double() - reference_apg(guided, pos).double()).abs().max().item(), 1e-4)

    def test_failed_probe_means_float32_and_is_cached(self):
        calls = []

        def no_fp64(*args, **kwargs):
            calls.append(1)
            raise TypeError("Cannot convert a MPS Tensor to float64 dtype")   # what MPS raises

        with patch.object(torch, "ones", no_fp64):
            for _ in range(3):
                self.assertEqual(core._correction_dtype(torch.zeros(1)), torch.float32)
        self.assertEqual(len(calls), 1)

    def test_env_override(self):
        os.environ["SNAPREF_FP64"] = "0"
        self.assertEqual(core._correction_dtype(torch.zeros(1)), torch.float32)
        os.environ["SNAPREF_FP64"] = "1"
        self.assertEqual(core._correction_dtype(torch.zeros(1)), torch.float64)


if __name__ == "__main__":
    unittest.main()
