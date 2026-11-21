"""MKM CUDA runtime policy tests; no physical GPU or printer required."""
import os
import sys
import unittest
from unittest.mock import patch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "v3"))
from mkm_gpu_runtime import GPURequiredError, choose_batch, require_cuda, cuda_batch


class FakeCuda:
    def __init__(self, available=True, free=12 * 1024**3, total=12 * 1024**3):
        self.available, self.free, self.total = available, free, total

    def is_available(self):
        return self.available

    def mem_get_info(self):
        return self.free, self.total


class FakeTorch:
    def __init__(self, **kwargs):
        self.cuda = FakeCuda(**kwargs)


class RuntimeTests(unittest.TestCase):
    def test_cuda_is_the_standard_execution_path(self):
        self.assertEqual(require_cuda(torch_module=FakeTorch()), "cuda")

    def test_unavailable_cuda_fails_without_cpu_fallback(self):
        with self.assertRaisesRegex(GPURequiredError, "CUDA unavailable"):
            require_cuda(torch_module=FakeTorch(available=False))

    def test_cpu_request_fails_in_production(self):
        with patch.dict(os.environ, {}, clear=True):
            with self.assertRaisesRegex(GPURequiredError, "no CPU fallback"):
                require_cuda(device="cpu", torch_module=FakeTorch())

    def test_test_fixture_cpu_requires_explicit_sandbox_mode(self):
        with patch.dict(os.environ, {"MKM_SYNTHETIC_TEST_ONLY": "1"}):
            self.assertEqual(require_cuda(device="cpu", torch_module=FakeTorch(available=False)), "cpu")

    def test_cuda_batch_respects_requested_limit(self):
        self.assertEqual(cuda_batch(8, width=6480, height=3600, torch_module=FakeTorch()), 8)

    def test_cuda_batch_shrinks_for_real_vram(self):
        t = FakeTorch(free=3 * 1024**3)
        n = cuda_batch(64, width=6480, height=3600, torch_module=t)
        self.assertGreaterEqual(n, 1)
        self.assertLess(n, 64)

    def test_low_memory_fails_closed(self):
        with self.assertRaises(GPURequiredError):
            choose_batch(64, width=6480, height=3600, available_bytes=600 * 1024**2)

    def test_batch_math_is_deterministic(self):
        kw = dict(width=6480, height=3600, available_bytes=12 * 1024**3)
        self.assertEqual(choose_batch(64, **kw), choose_batch(64, **kw))
        self.assertEqual(choose_batch(5, **kw), 5)


if __name__ == "__main__":
    unittest.main()
