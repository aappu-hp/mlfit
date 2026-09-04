"""
Tests for the Advisor orchestrator — the core pipeline.

CONCEPT: Mocking
─────────────────
The Advisor calls detect_hardware() and analyze_model() internally.
In tests, we don't want to hit real hardware or network. We use
unittest.mock.patch to swap those functions with controlled fakes.

    with patch("mlfit.core.advisor.detect_hardware", return_value=fake_hw):
        ...

This replaces the real detect_hardware *inside advisor.py* for the
duration of the `with` block. The real function is untouched everywhere
else.

CONCEPT: Test boundaries
─────────────────────────
Unit tests own one thing at a time. test_advisor.py tests that the
Advisor *orchestrates* correctly — it picks the right strategies, sorts
them, checks if the model fits, etc. It does NOT re-test memory math
(test_memory_estimation.py owns that) or strategy logic (test_strategies.py).
"""
import unittest
from unittest.mock import patch, MagicMock
from mlfit.core.models import (
    ModelProfile, HardwareProfile, GPUInfo, RecommendResult
)
from mlfit.core.memory import estimate_weights_gb


# ── Shared test fixtures ───────────────────────────────────────────────────────

def _llama_8b() -> ModelProfile:
    m = ModelProfile(
        model_id="meta-llama/Llama-3-8B",
        model_type="llm",
        num_parameters=8.03e9,
        dtype="bfloat16",
        architecture="LlamaForCausalLM",
        max_context_length=8192,
        num_hidden_layers=32,
        num_attention_heads=32,
        num_key_value_heads=8,
        hidden_size=4096,
    )
    m.estimated_size_gb = estimate_weights_gb(m.num_parameters, m.dtype)
    return m


def _qwen_05b() -> ModelProfile:
    """A tiny model — useful for testing 'fits easily' scenarios."""
    m = ModelProfile(
        model_id="Qwen/Qwen2.5-0.5B",
        model_type="llm",
        num_parameters=0.5e9,
        dtype="bfloat16",
        architecture="Qwen2ForCausalLM",
        max_context_length=32768,
        num_hidden_layers=24,
        num_attention_heads=14,
        num_key_value_heads=2,
        hidden_size=896,
    )
    m.estimated_size_gb = estimate_weights_gb(m.num_parameters, m.dtype)
    return m


def _rtx4090() -> HardwareProfile:
    return HardwareProfile(
        gpus=[GPUInfo(0, "NVIDIA RTX 4090", 24.0, 22.4, (8, 9), False)],
        total_vram_gb=22.4, gpu_backend="cuda",
        cpu_cores=16, cpu_threads=32, ram_gb=64.0,
        has_avx=True, has_avx2=True, has_avx512=True, disk_free_gb=500.0,
    )


def _macbook_m3() -> HardwareProfile:
    return HardwareProfile(
        gpus=[GPUInfo(0, "Apple M3 Pro", 36.0, 28.0, (0, 0), True)],
        total_vram_gb=28.0, gpu_backend="mps",
        cpu_cores=12, cpu_threads=12, ram_gb=36.0,
        has_avx=False, has_avx2=False, has_avx512=False, disk_free_gb=400.0,
    )


def _cpu_only() -> HardwareProfile:
    return HardwareProfile(
        gpus=[], total_vram_gb=0.0, gpu_backend="none",
        cpu_cores=32, cpu_threads=64, ram_gb=256.0,
        has_avx=True, has_avx2=True, has_avx512=True, disk_free_gb=2000.0,
    )


# ── Helper: run advisor with injected hardware and model ──────────────────────

def _recommend(model: ModelProfile, hw: HardwareProfile, backend: str = "auto") -> RecommendResult:
    """
    Run advisor.recommend() with mocked hardware and model detection.

    The two patches intercept the lazy property calls inside Advisor so
    we never touch real hardware or the HuggingFace network.
    """
    from mlfit.core.advisor import Advisor
    advisor = Advisor(model_id=model.model_id)

    with patch.object(type(advisor), "hardware", new_callable=lambda: property(lambda self: hw)):
        with patch.object(type(advisor), "model", new_callable=lambda: property(lambda self: model)):
            return advisor.recommend(backend=backend, )


# ── Tests: strategy selection ─────────────────────────────────────────────────

class TestAdvisorStrategySelection(unittest.TestCase):

    def test_cuda_llm_returns_vllm_as_top(self):
        """On NVIDIA CUDA with a standard LLM, vLLM should rank first."""
        result = _recommend(_llama_8b(), _rtx4090())
        self.assertGreater(len(result.configs), 0)
        top_backend = result.configs[0][1].backend
        self.assertEqual(top_backend, "vllm",
                         f"Expected vllm on top for CUDA, got {top_backend}")

    def test_mps_llm_excludes_vllm_and_tgi(self):
        """Apple Silicon doesn't support CUDA backends."""
        result = _recommend(_llama_8b(), _macbook_m3())
        backends = [c[1].backend for c in result.configs]
        self.assertNotIn("vllm", backends, "vLLM should not appear on MPS")
        self.assertNotIn("tgi",  backends, "TGI should not appear on MPS")

    def test_mps_llm_has_ollama_as_top(self):
        """Ollama is the best choice on Apple Silicon."""
        result = _recommend(_llama_8b(), _macbook_m3())
        top = result.configs[0][1].backend
        self.assertEqual(top, "ollama",
                         f"Expected ollama on top for MPS, got {top}")

    def test_cpu_only_excludes_vllm_and_tgi(self):
        """CPU-only machines have no CUDA — vLLM/TGI shouldn't appear."""
        result = _recommend(_llama_8b(), _cpu_only())
        backends = [c[1].backend for c in result.configs]
        self.assertNotIn("vllm", backends)
        self.assertNotIn("tgi",  backends)

    def test_configs_sorted_by_score_descending(self):
        """Results must be ranked best → worst."""
        result = _recommend(_llama_8b(), _rtx4090())
        scores = [c[0].score for c in result.configs]
        self.assertEqual(scores, sorted(scores, reverse=True),
                         "Configs should be sorted by feasibility score descending")

    def test_force_backend_returns_only_that_backend(self):
        """--backend llamacpp should return exactly one strategy."""
        result = _recommend(_llama_8b(), _rtx4090(), backend="llamacpp")
        backends = [c[1].backend for c in result.configs]
        self.assertEqual(backends, ["llamacpp"])


# ── Tests: fits / headroom logic ──────────────────────────────────────────────

class TestAdvisorFitsLogic(unittest.TestCase):

    def test_small_model_fits_on_4090(self):
        result = _recommend(_qwen_05b(), _rtx4090())
        self.assertTrue(result.is_feasible)

    def test_8b_fits_on_24gb(self):
        result = _recommend(_llama_8b(), _rtx4090())
        self.assertTrue(result.is_feasible)

    def test_model_does_not_fit_on_tiny_vram(self):
        """A 16GB model shouldn't fit on a 10GB GPU."""
        tiny_hw = HardwareProfile(
            gpus=[GPUInfo(0, "RTX 3080 10GB", 10.0, 9.0, (8, 6), False)],
            total_vram_gb=9.0, gpu_backend="cuda",
            cpu_cores=8, cpu_threads=16, ram_gb=32.0,
            has_avx=True, has_avx2=True, has_avx512=False, disk_free_gb=200.0,
        )
        result = _recommend(_llama_8b(), tiny_hw)
        self.assertFalse(result.is_feasible)

    def test_result_has_best_property(self):
        result = _recommend(_llama_8b(), _rtx4090())
        self.assertIsNotNone(result.best)
        self.assertIsNotNone(result.best.command)

    def test_result_has_alternatives_property(self):
        result = _recommend(_llama_8b(), _rtx4090())
        self.assertIsInstance(result.alternatives, list)


# ── Tests: output contracts ───────────────────────────────────────────────────

class TestAdvisorOutputContracts(unittest.TestCase):
    """
    CONCEPT: Contract testing
    ──────────────────────────
    We're not testing implementation details here — we're testing the
    *contract* that the Advisor promises to callers. If you change the
    internals but keep the contract, these tests stay green.
    """

    def test_to_dict_has_required_keys(self):
        """The JSON output format must always have these top-level keys."""
        result = _recommend(_llama_8b(), _rtx4090())
        d = result.to_dict()
        required = {"mlfit_version", "timestamp", "hardware", "model",
                    "is_feasible", "configs", "quantized_alternatives"}
        self.assertEqual(required, required & d.keys())

    def test_config_dicts_have_required_keys(self):
        result = _recommend(_llama_8b(), _rtx4090())
        d = result.to_dict()
        for cfg in d["configs"]:
            self.assertIn("rank", cfg)
            self.assertIn("backend", cfg)
            self.assertIn("feasibility_score", cfg)
            self.assertIn("command", cfg)
            self.assertIn("params", cfg)

    def test_feasibility_scores_in_range(self):
        result = _recommend(_llama_8b(), _rtx4090())
        for score, _, _ in result.configs:
            self.assertGreaterEqual(score.score, 0.0)
            self.assertLessEqual(score.score, 1.0)

    def test_commands_are_non_empty_strings(self):
        result = _recommend(_llama_8b(), _rtx4090())
        for _, cfg, _ in result.configs:
            self.assertIsInstance(cfg.command, str)
            self.assertGreater(len(cfg.command), 10)

    def test_to_json_is_valid_json(self):
        import json
        result = _recommend(_llama_8b(), _rtx4090())
        raw = result.to_json()
        parsed = json.loads(raw)
        self.assertIn("configs", parsed)

    def test_vllm_command_contains_model_id(self):
        result = _recommend(_llama_8b(), _rtx4090())
        vllm_configs = [c for c in result.configs if c[1].backend == "vllm"]
        self.assertTrue(vllm_configs)
        cmd = vllm_configs[0][1].command
        self.assertIn("meta-llama/Llama-3-8B", cmd)

    def test_tgi_command_uses_docker(self):
        result = _recommend(_llama_8b(), _rtx4090())
        tgi_configs = [c for c in result.configs if c[1].backend == "tgi"]
        if tgi_configs:
            self.assertIn("docker run", tgi_configs[0][1].command)

    def test_hardware_dict_has_gpu_info(self):
        result = _recommend(_llama_8b(), _rtx4090())
        hw_dict = result.to_dict()["hardware"]
        self.assertIn("gpus", hw_dict)
        self.assertEqual(len(hw_dict["gpus"]), 1)
        self.assertIn("RTX 4090", hw_dict["gpus"][0]["name"])
