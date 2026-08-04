from mlfit.core.models import GPUInfo


def detect_cuda(gpuid: str = "all") -> list:
    """
    Detect NVIDIA CUDA GPUs and return their info.
    Uses torch.cuda for accurate VRAM reporting.
    """
    try:
        import torch
    except ImportError:
        # torch not installed — fall back to nvidia-smi parsing
        return _detect_cuda_via_smi(gpuid)

    if not torch.cuda.is_available():
        return []

    n = torch.cuda.device_count()

    if gpuid == "all":
        indices = list(range(n))
    else:
        indices = [int(i) for i in gpuid.split(",") if i.strip()]
        indices = [i for i in indices if i < n]

    gpus = []
    for idx in indices:
        props = torch.cuda.get_device_properties(idx)
        free_bytes, total_bytes = torch.cuda.mem_get_info(idx)

        gpus.append(GPUInfo(
            index=idx,
            name=props.name,
            vram_total_gb=props.total_memory / 1e9,
            vram_free_gb=free_bytes / 1e9,
            compute_capability=(props.major, props.minor),
            is_unified_memory=False,
        ))

    return gpus


def _detect_cuda_via_smi(gpuid: str = "all") -> list:
    """
    Fallback: parse nvidia-smi output when torch is not available.
    Returns GPU info with approximate VRAM values.
    """
    import subprocess
    import re

    try:
        result = subprocess.run(
            ["nvidia-smi",
             "--query-gpu=index,name,memory.total,memory.free",
             "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=5
        )
        if result.returncode != 0:
            return []
    except (FileNotFoundError, subprocess.TimeoutExpired):
        return []

    gpus = []
    for line in result.stdout.strip().splitlines():
        parts = [p.strip() for p in line.split(",")]
        if len(parts) < 4:
            continue
        idx = int(parts[0])
        name = parts[1]
        total_mb = float(parts[2])
        free_mb = float(parts[3])

        if gpuid != "all" and str(idx) not in gpuid.split(","):
            continue

        gpus.append(GPUInfo(
            index=idx,
            name=name,
            vram_total_gb=total_mb / 1024,
            vram_free_gb=free_mb / 1024,
            compute_capability=(0, 0),   # unknown without torch
            is_unified_memory=False,
        ))

    return gpus
