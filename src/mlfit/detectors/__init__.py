"""
Hardware detection entry point for mlfit.

Auto-detects GPU backend and builds a normalized HardwareProfile.
Never imports torch at module level — it's slow. All imports are lazy.
"""
import logging

from mlfit.core.models import HardwareProfile, GPUInfo

logger = logging.getLogger(__name__)


def detect_hardware(gpuid: str = "all") -> HardwareProfile:
    """
    Main entry point. Auto-detects backend and available devices.

    Args:
        gpuid: "all" | "0" | "0,1,2" — which GPUs to include

    Returns:
        HardwareProfile with normalized hardware information
    """
    backend = _detect_gpu_backend()
    logger.debug("Detected GPU backend: %s", backend)

    if backend == "cuda":
        from mlfit.detectors.cuda import detect_cuda
        gpus = detect_cuda(gpuid)
    elif backend == "mps":
        from mlfit.detectors.mps import detect_mps
        gpus = detect_mps()
    elif backend == "rocm":
        from mlfit.detectors.rocm import detect_rocm
        gpus = detect_rocm(gpuid)
    else:
        gpus = []

    from mlfit.detectors.cpu import detect_cpu
    cpu_info = detect_cpu()

    total_vram_gb = sum(g.vram_free_gb for g in gpus)
    logger.info(
        "Hardware profile built: %d GPU(s), total_vram_gb=%.1f, backend=%s",
        len(gpus),
        total_vram_gb,
        backend,
    )

    return HardwareProfile(
        gpus=gpus,
        total_vram_gb=total_vram_gb,
        gpu_backend=backend,
        **cpu_info,
    )


def _detect_gpu_backend() -> str:
    """
    Detect available GPU backend without loading torch.
    Returns: "cuda" | "rocm" | "mps" | "none"
    """
    import subprocess
    import platform

    try:
        result = subprocess.run(
            ["nvidia-smi"], capture_output=True, timeout=3
        )
        if result.returncode == 0 and b"ROCm" not in result.stdout:
            return "cuda"
    except (FileNotFoundError, subprocess.TimeoutExpired):
        pass

    if platform.system() == "Darwin" and platform.machine() == "arm64":
        return "mps"

    try:
        result = subprocess.run(
            ["rocm-smi"], capture_output=True, timeout=3
        )
        if result.returncode == 0:
            return "rocm"
    except (FileNotFoundError, subprocess.TimeoutExpired):
        pass

    return "none"


def get_tensor_parallel_size(selected_gpus: list, num_heads: int) -> int:
    """
    Compute the largest valid tensor_parallel_size for vLLM/TGI.
    Must be a power of 2 and must divide num_attention_heads evenly.
    """
    n = len(selected_gpus)
    while n > 1:
        if num_heads % n == 0:
            return n
        n //= 2
    return 1
