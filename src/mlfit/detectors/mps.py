from mlfit.core.models import GPUInfo


def detect_mps() -> list:
    """
    Detect Apple Silicon GPU.

    Key insight: on Apple Silicon, RAM and GPU memory are the SAME physical
    pool (unified memory). So vram_total_gb == system RAM. We subtract ~4GB
    for OS + background apps as a conservative estimate.
    """
    import psutil
    import subprocess
    import json as _json

    total_ram_gb = psutil.virtual_memory().total / 1e9
    available_ram_gb = psutil.virtual_memory().available / 1e9

    # Try to get the actual chip name (M1/M2/M3/M4 + Pro/Max/Ultra)
    gpu_name = _get_apple_chip_name()

    # Reserve ~4GB for OS + other apps; use available as the ceiling
    vram_free_gb = max(0.0, available_ram_gb - 4.0)

    return [GPUInfo(
        index=0,
        name=gpu_name,
        vram_total_gb=total_ram_gb,
        vram_free_gb=vram_free_gb,
        compute_capability=(0, 0),   # Not applicable for MPS
        is_unified_memory=True,      # CRITICAL: RAM = GPU memory
    )]


def _get_apple_chip_name() -> str:
    """
    Try to determine Apple chip name (e.g., 'Apple M3 Max') via system_profiler.
    Falls back to 'Apple Silicon GPU' on any error.
    """
    import subprocess
    import json

    try:
        result = subprocess.run(
            ["system_profiler", "SPHardwareDataType", "-json"],
            capture_output=True, timeout=5, text=True
        )
        if result.returncode == 0:
            data = json.loads(result.stdout)
            hw = data.get("SPHardwareDataType", [{}])[0]
            chip = hw.get("chip_type", "")
            if chip:
                return chip
    except Exception:
        pass

    return "Apple Silicon GPU"
