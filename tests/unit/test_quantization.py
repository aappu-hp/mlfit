import unittest
from unittest.mock import patch, MagicMock

from mlfit.core.memory import estimate_gguf_size_gb
from mlfit.core.models import HardwareProfile, GPUInfo
from mlfit.quantization.advisor import (
    find_quantized_alternatives,
    _find_gguf_alternatives,
    _find_awq_alternative,
    _find_gptq_alternative,
    _check_cpu_offload,
    _infer_param_billions,
)
from mlfit.quantization.quality import (
    get_quality_profile,
    quality_label,
    supported_backends,
    QualityProfile,
)
from mlfit.quantization.hf_search import (
    _name_matches,
    _extract_core_name,
    _split_model_id,
)


# ── shared test fixtures ──────────────────────────────────────────────────────

def _cuda_hw(vram_gb: float = 24.0, ram_gb: float = 64.0) -> HardwareProfile:
    return HardwareProfile(
        gpus=[GPUInfo(0, "NVIDIA RTX 4090", vram_gb, vram_gb - 0.5, (8, 9), False)],
        total_vram_gb=vram_gb,
        gpu_backend="cuda",
        cpu_cores=16, cpu_threads=32,
        ram_gb=ram_gb,
        has_avx=True, has_avx2=True, has_avx512=True,
        disk_free_gb=500.0,
    )


def _mps_hw(ram_gb: float = 36.0) -> HardwareProfile:
    return HardwareProfile(
        gpus=[GPUInfo(0, "Apple M3 Max", ram_gb, ram_gb - 4.0, (0, 0), True)],
        total_vram_gb=ram_gb,
        gpu_backend="mps",
        cpu_cores=16, cpu_threads=16,
        ram_gb=ram_gb,
        has_avx=False, has_avx2=False, has_avx512=False,
        disk_free_gb=400.0,
    )


def _no_gpu_hw(ram_gb: float = 64.0) -> HardwareProfile:
    return HardwareProfile(
        gpus=[],
        total_vram_gb=0.0,
        gpu_backend="none",
        cpu_cores=8, cpu_threads=16,
        ram_gb=ram_gb,
        has_avx=True, has_avx2=True, has_avx512=False,
        disk_free_gb=200.0,
    )


def _fake_hf_model(model_id: str) -> MagicMock:
    m = MagicMock()
    m.modelId = model_id
    return m


# ── QualityProfile ────────────────────────────────────────────────────────────

class TestQualityProfile(unittest.TestCase):
    def test_known_quant_returns_correct_label(self):
        profile = get_quality_profile("Q4_K_M")
        self.assertEqual(profile.label, "low")

    def test_q8_is_minimal_quality_loss(self):
        self.assertEqual(quality_label("Q8_0"), "minimal")

    def test_q2k_is_high_quality_loss(self):
        self.assertEqual(quality_label("Q2_K"), "high")

    def test_awq_backend_includes_vllm(self):
        backends = supported_backends("AWQ")
        self.assertIn("vLLM", backends)

    def test_gguf_backend_includes_llamacpp(self):
        backends = supported_backends("Q4_K_M")
        self.assertIn("llama.cpp", backends)

    def test_unknown_quant_returns_fallback_profile(self):
        profile = get_quality_profile("UNKNOWN_QUANT_XYZ")
        self.assertIsInstance(profile, QualityProfile)
        self.assertEqual(profile.label, "low")

    def test_gptq_has_gpu_backend(self):
        backends = supported_backends("GPTQ")
        self.assertIn("vLLM", backends)

    def test_quality_hierarchy_is_correct(self):
        order = ["Q8_0", "Q5_K_M", "Q4_K_M", "Q3_K_M", "Q2_K"]
        labels = [quality_label(q) for q in order]
        # Q8 and Q5 are minimal/low; Q3 and Q2 are medium/high
        self.assertIn(labels[0], ("minimal", "low"))
        self.assertIn(labels[-1], ("medium", "high"))


# ── InferParamBillions ────────────────────────────────────────────────────────

class TestInferParamBillions(unittest.TestCase):
    def test_7b_returns_7(self):
        self.assertAlmostEqual(_infer_param_billions("Llama-3-7B-Instruct"), 7.0, delta=0.5)

    def test_70b_returns_70(self):
        self.assertAlmostEqual(_infer_param_billions("Meta-Llama-3-70B-Instruct"), 70.0, delta=2.0)

    def test_fractional_0_5b(self):
        self.assertAlmostEqual(_infer_param_billions("Qwen2.5-0.5B-Instruct"), 0.5, delta=0.1)

    def test_unknown_defaults_to_7(self):
        self.assertEqual(_infer_param_billions("some-model-no-size-info"), 7.0)

    def test_13b_model(self):
        self.assertAlmostEqual(_infer_param_billions("Model-13B-Chat"), 13.0, delta=0.5)


# ── QuantizationFitLogic (pure memory math) ───────────────────────────────────

class TestQuantizationFitLogic(unittest.TestCase):
    def test_q4_7b_fits_in_8gb(self):
        size = estimate_gguf_size_gb("Q4_K_M", 7.0)
        self.assertLess(size, 8.0 * 0.90)

    def test_q8_7b_fits_in_10gb(self):
        size = estimate_gguf_size_gb("Q8_0", 7.0)
        self.assertLess(size, 10.0 * 0.90)

    def test_q4_70b_does_not_fit_in_24gb(self):
        size = estimate_gguf_size_gb("Q4_K_M", 70.0)
        self.assertGreater(size, 24.0)

    def test_higher_quant_is_larger(self):
        q4 = estimate_gguf_size_gb("Q4_K_M", 7.0)
        q8 = estimate_gguf_size_gb("Q8_0", 7.0)
        self.assertGreater(q8, q4)


# ── hf_search helpers ─────────────────────────────────────────────────────────

class TestHFSearchHelpers(unittest.TestCase):
    def test_split_model_id_with_slash(self):
        org, name = _split_model_id("meta-llama/Llama-3-8B")
        self.assertEqual(org, "meta-llama")
        self.assertEqual(name, "Llama-3-8B")

    def test_split_model_id_without_slash(self):
        org, name = _split_model_id("some-model")
        self.assertEqual(org, "unknown")
        self.assertEqual(name, "some-model")

    def test_extract_core_name_strips_instruct(self):
        core = _extract_core_name("Llama-3-8B-Instruct")
        self.assertNotIn("Instruct", core)
        self.assertIn("Llama", core)

    def test_extract_core_name_strips_chat(self):
        core = _extract_core_name("Mistral-7B-Chat")
        self.assertNotIn("Chat", core)

    def test_name_matches_case_insensitive(self):
        self.assertTrue(_name_matches("Llama-3-8B", "bartowski/Llama-3-8B-GGUF"))

    def test_name_matches_returns_false_for_different_model(self):
        self.assertFalse(_name_matches("Llama-3-70B", "bartowski/Mistral-7B-GGUF"))


# ── find_gguf_alternatives ────────────────────────────────────────────────────

class TestFindGGUFAlternatives(unittest.TestCase):
    def _run(self, model_id, param_b, available_gb, fake_repos=None):
        fake_repos = fake_repos or ["bartowski/TestModel-GGUF"]
        with patch("mlfit.quantization.hf_search.list_models", return_value=iter([])):
            with patch("mlfit.quantization.hf_search.find_gguf_repos", return_value=fake_repos):
                return _find_gguf_alternatives(model_id, param_b, available_gb)

    def test_only_fitting_quants_returned(self):
        alts = self._run("meta-llama/Llama-3-8B", 8.0, 5.0)
        for alt in alts:
            self.assertLessEqual(alt.size_gb, 5.0 * 0.95)

    def test_sorted_by_size_ascending(self):
        alts = self._run("meta-llama/Llama-3-8B", 8.0, 24.0)
        sizes = [a.size_gb for a in alts]
        self.assertEqual(sizes, sorted(sizes))

    def test_quant_type_is_set(self):
        alts = self._run("meta-llama/Llama-3-8B", 8.0, 24.0)
        for alt in alts:
            self.assertIsNotNone(alt.quant_type)

    def test_backend_contains_llamacpp(self):
        alts = self._run("meta-llama/Llama-3-8B", 8.0, 24.0)
        for alt in alts:
            self.assertIn("llama.cpp", alt.backend)

    def test_uses_first_repo_from_search(self):
        alts = self._run(
            "meta-llama/Llama-3-8B", 8.0, 24.0,
            fake_repos=["bartowski/Llama-3-8B-GGUF"],
        )
        self.assertTrue(any("bartowski" in a.model_id for a in alts))


# ── find_awq_alternative ──────────────────────────────────────────────────────

class TestFindAWQAlternative(unittest.TestCase):
    def test_returns_none_when_too_large(self):
        with patch("mlfit.quantization.hf_search.list_models", return_value=iter([])):
            result = _find_awq_alternative("meta-llama/Llama-3-70B", 70.0, 5.0)
        self.assertIsNone(result)

    def test_returns_alternative_when_fits(self):
        fake = [_fake_hf_model("meta-llama/Llama-3-8B-AWQ")]
        with patch("mlfit.quantization.hf_search.list_models", return_value=iter(fake)):
            result = _find_awq_alternative("meta-llama/Llama-3-8B", 8.0, 24.0)
        self.assertIsNotNone(result)
        self.assertEqual(result.quant_type, "AWQ")

    def test_awq_quality_label_is_minimal(self):
        fake = [_fake_hf_model("meta-llama/Llama-3-8B-AWQ")]
        with patch("mlfit.quantization.hf_search.list_models", return_value=iter(fake)):
            result = _find_awq_alternative("meta-llama/Llama-3-8B", 8.0, 24.0)
        self.assertIn("minimal", result.quality_loss)


# ── find_gptq_alternative ─────────────────────────────────────────────────────

class TestFindGPTQAlternative(unittest.TestCase):
    def test_returns_none_when_too_large(self):
        with patch("mlfit.quantization.hf_search.list_models", return_value=iter([])):
            result = _find_gptq_alternative("meta-llama/Llama-3-70B", 70.0, 5.0)
        self.assertIsNone(result)

    def test_returns_alternative_when_fits(self):
        fake = [_fake_hf_model("TheBloke/Llama-3-8B-GPTQ")]
        with patch("mlfit.quantization.hf_search.list_models", return_value=iter(fake)):
            result = _find_gptq_alternative("meta-llama/Llama-3-8B", 8.0, 24.0)
        self.assertIsNotNone(result)
        self.assertEqual(result.quant_type, "GPTQ")

    def test_gptq_backend_includes_vllm(self):
        fake = [_fake_hf_model("TheBloke/Llama-3-8B-GPTQ")]
        with patch("mlfit.quantization.hf_search.list_models", return_value=iter(fake)):
            result = _find_gptq_alternative("meta-llama/Llama-3-8B", 8.0, 24.0)
        self.assertIn("vLLM", result.backend)


# ── check_cpu_offload ─────────────────────────────────────────────────────────

class TestCheckCPUOffload(unittest.TestCase):
    def test_feasible_when_model_fits_across_vram_and_ram(self):
        hw = _cuda_hw(vram_gb=10.0, ram_gb=64.0)
        # 8B bfloat16 ~ 16 GB. 10 VRAM + 64*0.6=38.4 RAM = 48.4 total → fits
        result = _check_cpu_offload("meta-llama/Llama-3-8B", 8.0, hw)
        self.assertIsNotNone(result)

    def test_infeasible_when_model_too_large_for_combined(self):
        hw = _cuda_hw(vram_gb=4.0, ram_gb=8.0)
        # 70B bfloat16 ~ 140 GB. 4 + 8*0.6=4.8 = 8.8 total → does not fit
        result = _check_cpu_offload("meta-llama/Llama-3-70B", 70.0, hw)
        self.assertIsNone(result)

    def test_cpu_offload_has_no_quant_type(self):
        hw = _cuda_hw(vram_gb=10.0, ram_gb=64.0)
        result = _check_cpu_offload("meta-llama/Llama-3-8B", 8.0, hw)
        if result:
            self.assertIsNone(result.quant_type)

    def test_cpu_offload_notes_mention_ngl(self):
        hw = _cuda_hw(vram_gb=10.0, ram_gb=64.0)
        result = _check_cpu_offload("meta-llama/Llama-3-8B", 8.0, hw)
        if result:
            self.assertIn("ngl", result.notes)


# ── find_quantized_alternatives (orchestrator) ───────────────────────────────

class TestFindQuantizedAlternatives(unittest.TestCase):
    def _run(self, model_id, hw, fake_search=None):
        fake_search = fake_search or []
        with patch("mlfit.quantization.hf_search.list_models", return_value=iter(fake_search)):
            return find_quantized_alternatives(model_id, hw)

    def test_returns_list(self):
        result = self._run("meta-llama/Llama-3-8B", _cuda_hw(24.0))
        self.assertIsInstance(result, list)

    def test_sorted_by_size_ascending(self):
        result = self._run("meta-llama/Llama-3-8B", _cuda_hw(24.0))
        sizes = [a.size_gb for a in result]
        self.assertEqual(sizes, sorted(sizes))

    def test_capped_at_max_alternatives(self):
        result = self._run("meta-llama/Llama-3-8B", _cuda_hw(100.0))
        self.assertLessEqual(len(result), 8)

    def test_no_awq_or_gptq_on_mps(self):
        result = self._run("Qwen/Qwen2.5-7B", _mps_hw())
        quant_types = [a.quant_type for a in result if a.quant_type]
        self.assertNotIn("AWQ", quant_types)
        self.assertNotIn("GPTQ", quant_types)

    def test_no_awq_or_gptq_on_cpu_only(self):
        result = self._run("meta-llama/Llama-3-8B", _no_gpu_hw())
        quant_types = [a.quant_type for a in result if a.quant_type]
        self.assertNotIn("AWQ", quant_types)
        self.assertNotIn("GPTQ", quant_types)

    def test_tiny_available_vram_returns_empty_or_few(self):
        result = self._run("meta-llama/Llama-3-70B", _cuda_hw(vram_gb=1.0, ram_gb=8.0))
        for alt in result:
            if alt.quant_type:
                self.assertLessEqual(alt.size_gb, 1.0)

    def test_uses_hf_search_result_for_gguf_id(self):
        fake = [_fake_hf_model("bartowski/Llama-3-8B-GGUF")]
        with patch("mlfit.quantization.hf_search.list_models", return_value=iter(fake)):
            result = find_quantized_alternatives("meta-llama/Llama-3-8B", _cuda_hw(24.0))
        gguf_ids = [a.model_id for a in result if a.quant_type and a.quant_type.startswith("Q")]
        self.assertTrue(
            any("bartowski" in mid for mid in gguf_ids),
            f"Expected bartowski in GGUF alternatives, got: {gguf_ids}",
        )
