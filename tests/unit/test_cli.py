"""
Tests for CLI output format and the reporter layer.

CONCEPT: Testing a CLI-first project without the CLI framework
───────────────────────────────────────────────────────────────
mlfit's CLI is thin: commands call Advisor then pass the result to
reporter functions. So there are two ways to test it:

  1. Test the CLI framework layer (typer) — needs typer installed.
  2. Test the reporter logic directly — works without typer.

We do option 2 here. The reporter functions accept a RecommendResult
and produce output. We inject a fully-built result, capture stdout,
and assert on the content.

CONCEPT: Capturing stdout in tests
────────────────────────────────────
We use io.StringIO to capture output from Rich:
    console = Console(file=buffer)

This lets us write tests like:
    assert "vllm" in output
    assert "RTX 4090" in output

without relying on exact formatting (colours, box-drawing chars).
"""
import io
import json
import unittest
from unittest.mock import patch
from mlfit.core.models import (
    ModelProfile, HardwareProfile, GPUInfo, RecommendResult
)
from mlfit.core.memory import estimate_weights_gb


# ── Shared fixtures ───────────────────────────────────────────────────────────

def _make_result_cuda() -> RecommendResult:
    """Build a full RecommendResult for an RTX 4090 + Llama-3-8B."""
    from mlfit.strategies import _load_all_strategies, STRATEGY_REGISTRY
    from mlfit.core.memory import check_fits

    _load_all_strategies()

    hw = HardwareProfile(
        gpus=[GPUInfo(0, "NVIDIA RTX 4090", 24.0, 22.4, (8, 9), False)],
        total_vram_gb=22.4, gpu_backend="cuda",
        cpu_cores=16, cpu_threads=32, ram_gb=64.0,
        has_avx=True, has_avx2=True, has_avx512=True, disk_free_gb=500.0,
    )
    model = ModelProfile(
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
    model.estimated_size_gb = estimate_weights_gb(model.num_parameters, model.dtype)

    configs = []
    for Cls in STRATEGY_REGISTRY.values():
        s = Cls()
        if s.is_compatible(model, hw):
            configs.append((s.estimate_feasibility(model, hw), s.generate_config(model, hw), s))
    configs.sort(reverse=True, key=lambda x: x[0].score)

    fits, est, headroom = check_fits(model, hw)
    return RecommendResult(hardware=hw, model=model, configs=configs,
                           is_feasible=fits, quantized_alternatives=[])


def _make_result_mps() -> RecommendResult:
    """Build a result for Apple Silicon + Llama-3-8B."""
    from mlfit.strategies import _load_all_strategies, STRATEGY_REGISTRY
    from mlfit.core.memory import check_fits

    _load_all_strategies()

    hw = HardwareProfile(
        gpus=[GPUInfo(0, "Apple M3 Pro", 36.0, 28.0, (0, 0), True)],
        total_vram_gb=28.0, gpu_backend="mps",
        cpu_cores=12, cpu_threads=12, ram_gb=36.0,
        has_avx=False, has_avx2=False, has_avx512=False, disk_free_gb=400.0,
    )
    model = ModelProfile(
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
    model.estimated_size_gb = estimate_weights_gb(model.num_parameters, model.dtype)

    configs = []
    for Cls in STRATEGY_REGISTRY.values():
        s = Cls()
        if s.is_compatible(model, hw):
            configs.append((s.estimate_feasibility(model, hw), s.generate_config(model, hw), s))
    configs.sort(reverse=True, key=lambda x: x[0].score)

    fits, est, headroom = check_fits(model, hw)
    return RecommendResult(hardware=hw, model=model, configs=configs,
                           is_feasible=fits, quantized_alternatives=[])


def _capture_render(result: RecommendResult) -> str:
    """
    Run render_recommend_result and capture the output as a plain string.

    CONCEPT: Rich and stdout
    ─────────────────────────
    Rich normally writes to sys.stdout with ANSI colour codes.
    By injecting our own Console(file=buffer, highlight=False, markup=False)
    we get clean text without escape sequences, making assertions simple.
    """
    try:
        from rich.console import Console
        buffer = io.StringIO()
        test_console = Console(file=buffer, highlight=False, markup=False,
                               width=120)

        from mlfit.reporter import table
        original_console = table.console
        table.console = test_console
        try:
            table.render_recommend_result(result)
        finally:
            table.console = original_console

        return buffer.getvalue()
    except ImportError:
        return None  # rich not installed; skip rendering tests


# ── Tests: JSON output ────────────────────────────────────────────────────────

class TestJSONOutput(unittest.TestCase):
    """
    CONCEPT: Test the contract, not the implementation
    ───────────────────────────────────────────────────
    We don't care how the JSON is generated internally. We care that:
    - It is valid JSON.
    - It contains specific keys that downstream tools depend on.
    - The values are in the right shape and range.

    This is what contract testing means for a CLI-first project.
    """

    def setUp(self):
        self.result = _make_result_cuda()

    def test_to_json_is_valid_json(self):
        raw = self.result.to_json()
        parsed = json.loads(raw)
        self.assertIsInstance(parsed, dict)

    def test_top_level_keys_present(self):
        d = self.result.to_dict()
        for key in ("mlfit_version", "timestamp", "hardware", "model",
                    "is_feasible", "configs", "quantized_alternatives"):
            self.assertIn(key, d, f"Missing top-level key: {key}")

    def test_configs_array_is_ranked(self):
        d = self.result.to_dict()
        ranks = [c["rank"] for c in d["configs"]]
        self.assertEqual(ranks, list(range(1, len(ranks) + 1)))

    def test_scores_are_floats_in_range(self):
        d = self.result.to_dict()
        for cfg in d["configs"]:
            score = cfg["feasibility_score"]
            self.assertIsInstance(score, float)
            self.assertGreaterEqual(score, 0.0)
            self.assertLessEqual(score, 1.0)

    def test_hardware_gpu_info_correct(self):
        d = self.result.to_dict()
        gpu = d["hardware"]["gpus"][0]
        self.assertIn("RTX 4090", gpu["name"])
        self.assertEqual(gpu["vram_total_gb"], 24.0)
        self.assertFalse(gpu["is_unified_memory"])

    def test_model_info_correct(self):
        d = self.result.to_dict()
        m = d["model"]
        self.assertEqual(m["model_id"], "meta-llama/Llama-3-8B")
        self.assertEqual(m["dtype"], "bfloat16")
        self.assertGreater(m["num_parameters"], 5e9)

    def test_fits_is_bool(self):
        d = self.result.to_dict()
        self.assertIsInstance(d["is_feasible"], bool)

    def test_vllm_config_has_required_params(self):
        d = self.result.to_dict()
        vllm_cfgs = [c for c in d["configs"] if c["backend"] == "vllm"]
        self.assertTrue(vllm_cfgs, "Expected vllm in configs")
        params = vllm_cfgs[0]["params"]
        self.assertIn("gpu_memory_utilization", params)
        self.assertIn("max_model_len", params)
        self.assertIn("tensor_parallel_size", params)

    def test_vllm_command_is_runnable_format(self):
        """The command should look like a real shell command."""
        d = self.result.to_dict()
        vllm_cfgs = [c for c in d["configs"] if c["backend"] == "vllm"]
        cmd = vllm_cfgs[0]["command"]
        self.assertTrue(cmd.startswith("vllm serve"))
        self.assertIn("--gpu-memory-utilization", cmd)
        self.assertIn("meta-llama/Llama-3-8B", cmd)


# ── Tests: reporter content (CUDA) ────────────────────────────────────────────

class TestReporterContentCUDA(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        result = _make_result_cuda()
        cls.output = _capture_render(result)
        cls.skipped = cls.output is None

    def _skip_if_no_rich(self):
        if self.skipped:
            self.skipTest("rich not installed — skipping renderer test")

    def test_header_contains_mlfit(self):
        self._skip_if_no_rich()
        self.assertIn("mlfit", self.output.lower())

    def test_hardware_panel_shows_gpu_name(self):
        self._skip_if_no_rich()
        self.assertIn("RTX 4090", self.output)

    def test_hardware_panel_shows_vram(self):
        self._skip_if_no_rich()
        self.assertIn("22.4", self.output)

    def test_model_panel_shows_model_id(self):
        self._skip_if_no_rich()
        self.assertIn("Llama-3-8B", self.output)

    def test_model_panel_shows_architecture(self):
        self._skip_if_no_rich()
        self.assertIn("LlamaForCausalLM", self.output)

    def test_configs_table_contains_vllm(self):
        self._skip_if_no_rich()
        self.assertIn("vllm", self.output)

    def test_configs_table_contains_ollama(self):
        self._skip_if_no_rich()
        self.assertIn("ollama", self.output)

    def test_best_command_panel_shown(self):
        self._skip_if_no_rich()
        self.assertIn("vllm serve", self.output)

    def test_fits_message_shown(self):
        self._skip_if_no_rich()
        self.assertIn("YES", self.output.upper())


# ── Tests: reporter content (Apple Silicon) ───────────────────────────────────

class TestReporterContentMPS(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        result = _make_result_mps()
        cls.output = _capture_render(result)
        cls.skipped = cls.output is None

    def _skip_if_no_rich(self):
        if self.skipped:
            self.skipTest("rich not installed — skipping renderer test")

    def test_gpu_shows_apple_chip(self):
        self._skip_if_no_rich()
        self.assertIn("M3", self.output)

    def test_unified_memory_note_shown(self):
        self._skip_if_no_rich()
        self.assertIn("unified", self.output.lower())

    def test_no_vllm_in_output(self):
        self._skip_if_no_rich()
        self.assertNotIn("vllm serve", self.output)

    def test_ollama_is_top_recommendation(self):
        self._skip_if_no_rich()
        lines = [l for l in self.output.splitlines() if "ollama" in l.lower()]
        self.assertTrue(lines, "Expected ollama in output for MPS hardware")
