from mlfit.core.models import GPUInfo


def detect_rocm(gpuid: str = "all") -> list:
    """
    Detect AMD GPUs via ROCm.

    ROCm presents as CUDA-compatible to PyTorch.
    We detect the ROCm-specific build via torch.version.hip.
    """
    try:
        import torch
    except ImportError:
        return _detect_rocm_via_smi(gpuid)

    if not torch.cuda.is_available():
        return []

    n = torch.cuda.device_count()
    indices = list(range(n)) if gpuid == "all" else [
        int(i) for i in gpuid.split(",") if i.strip()
    ]
    indices = [i for i in indices if i < n]

    gpus = []
    for idx in indices:
        props = torch.cuda.get_device_properties(idx)
        free_bytes, _ = torch.cuda.mem_get_info(idx)

        gpus.append(GPUInfo(
            index=idx,
            name=props.name,
            vram_total_gb=props.total_memory / 1e9,
            vram_free_gb=free_bytes / 1e9,
            compute_capability=(props.major, props.minor),
            is_unified_memory=False,
        ))

    return gpus


def _detect_rocm_via_smi(gpuid: str = "all") -> list:
    """Fallback: parse rocm-smi when torch is unavailable."""
    import subprocess

    try:
        result = subprocess.run(
            ["rocm-smi", "--showmeminfo", "vram", "--json"],
            capture_output=True, text=True, timeout=5
        )
        if result.returncode != 0:
            return []
    except (FileNotFoundError, subprocess.TimeoutExpired):
        return []

    gpus = []
    try:
        import json
        data = json.loads(result.stdout)
        for card_key, info in data.items():
            idx_str = card_key.replace("card", "")
            if not idx_str.isdigit():
                continue
            idx = int(idx_str)
            if gpuid != "all" and str(idx) not in gpuid.split(","):
                continue

            total_bytes = int(info.get("VRAM Total Memory (B)", 0))
            used_bytes = int(info.get("VRAM Total Used Memory (B)", 0))
            free_bytes = max(0, total_bytes - used_bytes)

            gpus.append(GPUInfo(
                index=idx,
                name=f"AMD GPU {idx}",
                vram_total_gb=total_bytes / 1e9,
                vram_free_gb=free_bytes / 1e9,
                compute_capability=(0, 0),
                is_unified_memory=False,
            ))
    except Exception:
        pass

    return gpus
