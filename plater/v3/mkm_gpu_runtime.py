"""Mandatory CUDA policy and conservative, measured-VRAM batch planning.

The production MKM Plater does not quietly fall back to CPU for slicing or
rating. An explicit synthetic fixture mode exists solely for offline tests.
No hardware action or printer communication lives in this module.
"""
import os


class GPURequiredError(RuntimeError):
    """Cannot execute production slicing or rating without usable CUDA."""


def require_cuda(device="cuda", torch_module=None):
    """Validate the requested engine; return the resolved torch device name.

    'cpu' is permitted only for isolated synthetic fixture tests invoked with
    MKM_SYNTHETIC_TEST_ONLY=1. It is NEVER a production fallback.
    """
    if device == "cpu" and os.environ.get("MKM_SYNTHETIC_TEST_ONLY") == "1":
        return "cpu"
    if device != "cuda":
        raise GPURequiredError("MKM production plater requires CUDA; no CPU fallback (requested %r)" % device)
    if torch_module is None:
        try:
            import torch as torch_module
        except ImportError as e:
            raise GPURequiredError("CUDA PyTorch is required for MKM GPU plater") from e
    try:
        if not torch_module.cuda.is_available():
            raise GPURequiredError("CUDA unavailable; stopped before slicing or rating")
        free_bytes, total_bytes = torch_module.cuda.mem_get_info()
        if min(int(free_bytes), int(total_bytes)) <= 0:
            raise GPURequiredError("CUDA reported zero usable VRAM")
    except GPURequiredError:
        raise
    except Exception as e:
        raise GPURequiredError("CUDA device initialization failed: %s" % e) from e
    return "cuda"


def choose_batch(requested, *, width, height, available_bytes,
                 temporary_copies=12, use_fraction=0.30, reserve_bytes=512 * 1024**2):
    """Choose a positive batch count from measured free VRAM.

    This memory estimate is a resource heuristic, not print physics.
    It budgets temporary full-resolution allocations, rather than just the
    output masks. Too little memory raises instead of silently moving to CPU.
    """
    if int(requested) < 1 or int(width) < 1 or int(height) < 1 or int(temporary_copies) < 1:
        raise ValueError("batch, image dimensions and temporary_copies must be positive")
    if not (0.0 < use_fraction <= 1.0):
        raise ValueError("use_fraction must be in (0, 1]")
    if int(available_bytes) <= int(reserve_bytes):
        raise GPURequiredError("Insufficient free VRAM after the MKM reserve")
    available = int((int(available_bytes) - int(reserve_bytes)) * use_fraction)
    per_layer = int(width) * int(height) * int(temporary_copies)
    n = available // per_layer
    if n < 1:
        raise GPURequiredError("Insufficient VRAM for a single layer at printer resolution")
    return min(int(requested), n)


def cuda_batch(requested, *, width, height, temporary_copies=12, torch_module=None):
    """Production VRAM-derived layer batch; no automatic CPU fallback."""
    require_cuda(torch_module=torch_module)
    if torch_module is None:
        import torch as torch_module
    free_bytes, _ = torch_module.cuda.mem_get_info()
    return choose_batch(requested, width=width, height=height,
                        available_bytes=free_bytes, temporary_copies=temporary_copies)
