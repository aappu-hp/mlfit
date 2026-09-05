from dataclasses import dataclass

from mlfit.core.models import HardwareProfile, GPUInfo
from mlfit.detectors.cpu import detect_cpu
from mlfit.tui.widgets.hardware_panel import HardwareBar


def test_detect_cpu_reports_available_ram_and_swap():
    info = detect_cpu()
    assert info["ram_available_gb"] > 0
    assert info["ram_available_gb"] <= info["ram_gb"]
    assert info["swap_total_gb"] >= 0
    assert info["swap_used_gb"] >= 0


def test_gpu_api_labels():
    assert HardwareBar._gpu_api("mps", (0, 0)) == "Metal"
    assert HardwareBar._gpu_api("cuda", (8, 9)) == "CUDA 8.9"
    assert HardwareBar._gpu_api("cuda", (0, 0)) == "CUDA"
    assert HardwareBar._gpu_api("rocm", (0, 0)) == "ROCm"
    assert HardwareBar._gpu_api("none", (0, 0)) == "CPU"


def _unified_profile() -> HardwareProfile:
    gpu = GPUInfo(
        index=0,
        name="Apple M4",
        vram_total_gb=17.0,
        vram_free_gb=0.0,
        compute_capability=(0, 0),
        is_unified_memory=True,
    )
    return HardwareProfile(
        gpus=[gpu],
        total_vram_gb=0.0,
        gpu_backend="mps",
        cpu_cores=10,
        cpu_threads=10,
        ram_gb=17.0,
        has_avx=False,
        has_avx2=False,
        has_avx512=False,
        disk_free_gb=210.0,
        ram_available_gb=5.1,
        swap_total_gb=4.0,
        swap_used_gb=2.1,
    )


def test_unified_memory_shows_shared_and_available_ram():
    bar = HardwareBar()
    bar._hw = _unified_profile()

    compute = bar._compute_line()
    assert "17GB shared" in compute
    assert "Metal" in compute
    assert "0.0/" not in compute        # the misleading unified "free" is gone

    memory = bar._memory_line()
    assert "5.1/17GB" in memory          # available/total RAM (the bug fix)
    assert "Swap:" in memory


def test_swap_omitted_when_absent():
    profile = _unified_profile()
    profile.swap_total_gb = 0.0
    bar = HardwareBar()
    bar._hw = profile
    assert "Swap:" not in bar._memory_line()
