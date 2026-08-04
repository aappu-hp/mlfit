"""Unit tests for memory estimation math — no GPU required."""
import unittest
from mlfit.core.memory import (
    estimate_weights_gb,
    estimate_vram_gb,
    estimate_gguf_size_gb,
    check_fits,
    BYTES_PER_PARAM,
)
from mlfit.core.models import ModelProfile, HardwareProfile, GPUInfo


# ── Fixtures ──────────────────────────────────────────────────────────────────

def make_llama_8b() -> ModelProfile:
    """Realistic Llama-3-8B profile."""
    m = ModelProfile(
        model_id="meta-llama/Llama-3-8B",
        model_type="llm",
        num_parameters=8.03e9,
        dtype="bfloat16",
        architecture="LlamaForCausalLM",
        max_context_length=8192,
        num_hidden_layers=32,
        num_attention_heads=32,
        num_key_value_heads=8,   # GQA
        hidden_size=4096,
    )
    m.estimated_size_gb = estimate_weights_gb(m.num_parameters, m.dtype)
    return m


def make_rtx_4090() -> HardwareProfile:
    return HardwareProfile(
        gpus=[GPUInfo(0, "NVIDIA RTX 4090", 24.0, 22.4, (8, 9), False)],
        total_vram_gb=22.4, gpu_backend="cuda",
        cpu_cores=16, cpu_threads=32, ram_gb=64.0,
        has_avx=True, has_avx2=True, has_avx512=True, disk_free_gb=500.0,
    )


def make_cpu_only() -> HardwareProfile:
    return HardwareProfile(
        gpus=[], total_vram_gb=0.0, gpu_backend="none",
        cpu_cores=32, cpu_threads=64, ram_gb=256.0,
        has_avx=True, has_avx2=True, has_avx512=True, disk_free_gb=2000.0,
    )


# ── Tests ─────────────────────────────────────────────────────────────────────

class TestWeightEstimation(unittest.TestCase):
    def test_bfloat16_8b(self):
        gb = estimate_weights_gb(8e9, "bfloat16")
        assert 14.0 < gb < 18.0, f"Expected ~16GB for 8B bfloat16, got {gb:.1f}"

    def test_float16_7b(self):
        gb = estimate_weights_gb(7e9, "float16")
        assert 12.0 < gb < 16.0, f"Expected ~14GB for 7B float16, got {gb:.1f}"

    def test_q4_7b(self):
        gb = estimate_weights_gb(7e9, "Q4_K_M")
        assert 3.0 < gb < 5.0, f"Expected ~4GB for 7B Q4_K_M, got {gb:.1f}"

    def test_int8_7b(self):
        gb = estimate_weights_gb(7e9, "int8")
        assert 6.0 < gb < 9.0, f"Expected ~7GB for 7B int8, got {gb:.1f}"

    def test_scaling_proportional(self):
        gb_7b = estimate_weights_gb(7e9, "float16")
        gb_14b = estimate_weights_gb(14e9, "float16")
        ratio = gb_14b / gb_7b
        assert 1.9 < ratio < 2.1, f"14B should be ~2x 7B, got ratio={ratio:.2f}"


class TestVRAMEstimation(unittest.TestCase):
    def test_llama_8b_on_4090(self):
        model = make_llama_8b()
        vram = estimate_vram_gb(model, max_seq_len=4096, max_batch=1)
        # Should be weights (~16GB) + small KV cache + overhead
        assert 15.0 < vram < 22.0, f"Expected 16-20GB for Llama-8B, got {vram:.1f}"

    def test_kv_cache_scales_with_seq_len(self):
        model = make_llama_8b()
        vram_short = estimate_vram_gb(model, max_seq_len=512, max_batch=1)
        vram_long = estimate_vram_gb(model, max_seq_len=8192, max_batch=1)
        assert vram_long > vram_short, "Longer context should use more VRAM"

    def test_kv_cache_scales_with_batch(self):
        model = make_llama_8b()
        vram_b1 = estimate_vram_gb(model, max_seq_len=4096, max_batch=1)
        vram_b16 = estimate_vram_gb(model, max_seq_len=4096, max_batch=16)
        assert vram_b16 > vram_b1, "Larger batch should use more VRAM"


class TestGGUFSizes(unittest.TestCase):
    def test_q4_7b(self):
        size = estimate_gguf_size_gb("Q4_K_M", 7.0)
        assert 3.5 < size < 5.0, f"Expected ~4.1GB for Q4_K_M 7B, got {size:.1f}"

    def test_q8_7b(self):
        size = estimate_gguf_size_gb("Q8_0", 7.0)
        assert 6.5 < size < 9.0, f"Expected ~7.7GB for Q8_0 7B, got {size:.1f}"

    def test_larger_model_is_bigger(self):
        size_7b = estimate_gguf_size_gb("Q4_K_M", 7.0)
        size_13b = estimate_gguf_size_gb("Q4_K_M", 13.0)
        assert size_13b > size_7b, "13B should be larger than 7B"


class TestCheckFits(unittest.TestCase):
    def test_8b_fits_on_4090(self):
        model = make_llama_8b()
        hw = make_rtx_4090()
        fits, estimated, headroom = check_fits(model, hw)
        assert fits, f"8B model should fit on 24GB 4090 (estimated {estimated:.1f}GB)"
        assert headroom > 0

    def test_8b_fits_on_cpu_with_enough_ram(self):
        model = make_llama_8b()
        hw = make_cpu_only()
        fits, estimated, headroom = check_fits(model, hw)
        assert fits, "8B model (16GB) should fit in 256GB RAM"
