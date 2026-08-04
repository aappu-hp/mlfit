def detect_cpu() -> dict:
    """
    Detect CPU properties, RAM, and AVX instruction set support.
    Returns a dict that unpacks directly into HardwareProfile fields.
    """
    import psutil

    # RAM and disk
    ram_gb = psutil.virtual_memory().total / 1e9
    disk_free_gb = psutil.disk_usage("/").free / 1e9
    cpu_cores = psutil.cpu_count(logical=False) or 1
    cpu_threads = psutil.cpu_count(logical=True) or cpu_cores

    # AVX support via py-cpuinfo
    has_avx, has_avx2, has_avx512 = _detect_avx()

    return {
        "cpu_cores": cpu_cores,
        "cpu_threads": cpu_threads,
        "ram_gb": ram_gb,
        "disk_free_gb": disk_free_gb,
        "has_avx": has_avx,
        "has_avx2": has_avx2,
        "has_avx512": has_avx512,
    }


def _detect_avx() -> tuple:
    """
    Detect AVX instruction set support.
    Returns (has_avx, has_avx2, has_avx512).
    """
    try:
        from cpuinfo import get_cpu_info
        info = get_cpu_info()
        flags = info.get("flags", [])
        has_avx = "avx" in flags
        has_avx2 = "avx2" in flags
        has_avx512 = any("avx512" in f for f in flags)
        return has_avx, has_avx2, has_avx512
    except Exception:
        pass

    # Fallback: try reading /proc/cpuinfo (Linux)
    try:
        with open("/proc/cpuinfo") as f:
            content = f.read().lower()
        has_avx = "avx" in content
        has_avx2 = "avx2" in content
        has_avx512 = "avx512" in content
        return has_avx, has_avx2, has_avx512
    except Exception:
        pass

    return False, False, False
