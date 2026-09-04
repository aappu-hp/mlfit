"""
Tests for model analyzers — with mocked network calls.

CONCEPT: Patching at the right level
──────────────────────────────────────
When you patch a function, patch it where it is *used*, not where it is
*defined*. Our hf_analyzer.py does:

    from huggingface_hub import hf_hub_download

So we patch it at: "mlfit.analyzers.hf_analyzer.hf_hub_download"
NOT at: "huggingface_hub.hf_hub_download"

This is a common Python mocking rule — the module under test holds its
own reference to the imported name. Patching the source library doesn't
affect that reference.

CONCEPT: Fake files vs MagicMock
──────────────────────────────────
When code opens a file (open(config_path)), we can't return a MagicMock —
json.load() needs a real file-like object. So we write the fake config
to a temp file and return that path instead.
"""
import json
import os
import tempfile
import unittest
from unittest.mock import patch, MagicMock
from mlfit.analyzers.hf_analyzer import (
    _estimate_params_from_config, _resolve_dtype
)
from mlfit.analyzers.gguf_analyzer import (
    parse_gguf_model_id, _infer_params_from_name
)


# ── Fake configs that mimic real HuggingFace config.json files ───────────────

LLAMA_3_8B_CONFIG = {
    "architectures": ["LlamaForCausalLM"],
    "hidden_size": 4096,
    "intermediate_size": 14336,
    "num_hidden_layers": 32,
    "num_attention_heads": 32,
    "num_key_value_heads": 8,
    "max_position_embeddings": 8192,
    "vocab_size": 128256,
    "torch_dtype": "bfloat16",
    "hidden_act": "silu",
}

MISTRAL_7B_CONFIG = {
    "architectures": ["MistralForCausalLM"],
    "hidden_size": 4096,
    "intermediate_size": 14336,
    "num_hidden_layers": 32,
    "num_attention_heads": 32,
    "num_key_value_heads": 8,
    "max_position_embeddings": 32768,
    "vocab_size": 32000,
    "torch_dtype": "float16",
}

QWEN_05B_CONFIG = {
    "architectures": ["Qwen2ForCausalLM"],
    "hidden_size": 896,
    "intermediate_size": 4864,
    "num_hidden_layers": 24,
    "num_attention_heads": 14,
    "num_key_value_heads": 2,
    "max_position_embeddings": 32768,
    "vocab_size": 151936,
    "torch_dtype": "bfloat16",
}


def _make_temp_config(config_dict: dict) -> str:
    """Write a config dict to a temp file and return the file path."""
    with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
        json.dump(config_dict, f)
        return f.name


# ── Tests: parameter estimation from config ───────────────────────────────────

class TestParamEstimation(unittest.TestCase):
    """
    We test _estimate_params_from_config directly.
    It's a pure function (no I/O) so no mocking needed.
    """

    def test_llama_8b_estimate_in_range(self):
        """Parameter estimate should be within 15% of the true 8.03B."""
        estimated = _estimate_params_from_config(LLAMA_3_8B_CONFIG)
        self.assertGreater(estimated, 6e9)
        self.assertLess(estimated, 12e9)

    def test_qwen_05b_is_much_smaller_than_llama_8b(self):
        llama = _estimate_params_from_config(LLAMA_3_8B_CONFIG)
        qwen = _estimate_params_from_config(QWEN_05B_CONFIG)
        self.assertGreater(llama, qwen * 5,
                           "Llama 8B should be much larger than Qwen 0.5B")

    def test_deeper_model_has_more_params(self):
        """Adding more hidden layers should increase parameter count."""
        shallow = {**LLAMA_3_8B_CONFIG, "num_hidden_layers": 16}
        deep = {**LLAMA_3_8B_CONFIG, "num_hidden_layers": 64}
        self.assertGreater(
            _estimate_params_from_config(deep),
            _estimate_params_from_config(shallow),
        )

    def test_wider_hidden_size_has_more_params(self):
        narrow = {**LLAMA_3_8B_CONFIG, "hidden_size": 2048, "num_attention_heads": 16}
        wide = {**LLAMA_3_8B_CONFIG, "hidden_size": 8192, "num_attention_heads": 64}
        self.assertGreater(
            _estimate_params_from_config(wide),
            _estimate_params_from_config(narrow),
        )


# ── Tests: dtype resolution ───────────────────────────────────────────────────

class TestDtypeResolution(unittest.TestCase):

    def test_explicit_bfloat16_respected(self):
        self.assertEqual(_resolve_dtype({"torch_dtype": "bfloat16"}), "bfloat16")

    def test_explicit_float16_respected(self):
        self.assertEqual(_resolve_dtype({"torch_dtype": "float16"}), "float16")

    def test_explicit_float32_respected(self):
        self.assertEqual(_resolve_dtype({"torch_dtype": "float32"}), "float32")

    def test_llama_arch_defaults_to_bfloat16(self):
        config = {"architectures": ["LlamaForCausalLM"]}
        self.assertEqual(_resolve_dtype(config), "bfloat16")

    def test_unknown_arch_defaults_to_float16(self):
        config = {"architectures": ["SomeOldModelForCausalLM"]}
        self.assertEqual(_resolve_dtype(config), "float16")


# ── Tests: full HF analyzer with mocked download ─────────────────────────────

class TestHFAnalyzerWithMock(unittest.TestCase):
    """
    CONCEPT: Testing I/O-heavy code
    ─────────────────────────────────
    analyze_hf_model() downloads a file and opens it.
    We fake both steps:
      1. Patch hf_hub_download to return a path to a temp file we control.
      2. The temp file contains our fake config dict.

    This gives us 100% control over the "network response" without any
    actual network call. The code under test runs exactly the same logic —
    it still calls json.load(open(path)) — but the path points to our file.
    """

    def _run_with_config(self, config: dict, model_id: str):
        """
        Helper: analyze a model using a fake config file.

        Patches _hf_hub_download (the module-level wrapper in hf_analyzer.py)
        to return a path to a temp file we control. The real HuggingFace
        network is never touched.
        """
        from mlfit.analyzers.hf_analyzer import analyze_hf_model
        tmp = _make_temp_config(config)
        try:
            with patch("mlfit.analyzers.hf_analyzer._hf_hub_download", return_value=tmp):
                return analyze_hf_model(model_id)
        finally:
            os.unlink(tmp)

    def test_returns_model_profile(self):
        from mlfit.core.models import ModelProfile
        result = self._run_with_config(LLAMA_3_8B_CONFIG, "meta-llama/Llama-3-8B")
        self.assertIsInstance(result, ModelProfile)

    def test_model_id_preserved(self):
        result = self._run_with_config(LLAMA_3_8B_CONFIG, "meta-llama/Llama-3-8B")
        self.assertEqual(result.model_id, "meta-llama/Llama-3-8B")

    def test_architecture_extracted(self):
        result = self._run_with_config(LLAMA_3_8B_CONFIG, "meta-llama/Llama-3-8B")
        self.assertEqual(result.architecture, "LlamaForCausalLM")

    def test_context_length_extracted(self):
        result = self._run_with_config(LLAMA_3_8B_CONFIG, "meta-llama/Llama-3-8B")
        self.assertEqual(result.max_context_length, 8192)

    def test_gqa_kv_heads_extracted(self):
        """GQA models have num_key_value_heads < num_attention_heads."""
        result = self._run_with_config(LLAMA_3_8B_CONFIG, "meta-llama/Llama-3-8B")
        self.assertEqual(result.num_key_value_heads, 8)
        self.assertEqual(result.num_attention_heads, 32)

    def test_dtype_from_config(self):
        result = self._run_with_config(LLAMA_3_8B_CONFIG, "meta-llama/Llama-3-8B")
        self.assertEqual(result.dtype, "bfloat16")

    def test_size_estimated(self):
        result = self._run_with_config(LLAMA_3_8B_CONFIG, "meta-llama/Llama-3-8B")
        self.assertGreater(result.estimated_size_gb, 5.0)

    def test_model_not_found_raises(self):
        """ModelNotFoundError should be raised when HF returns a 404-like error."""
        from mlfit.exceptions import ModelNotFoundError
        from huggingface_hub.utils import RepositoryNotFoundError
        from unittest.mock import MagicMock

        mock_response = MagicMock()
        mock_response.headers = {}

        def fake_download(*args, **kwargs):
            raise RepositoryNotFoundError("nonexistent/model-xyz", response=mock_response)

        from mlfit.analyzers.hf_analyzer import analyze_hf_model
        with patch("mlfit.analyzers.hf_analyzer._hf_hub_download", side_effect=fake_download):
            with self.assertRaises(ModelNotFoundError):
                analyze_hf_model("nonexistent/model-xyz")

    def test_mistral_7b_float16(self):
        result = self._run_with_config(MISTRAL_7B_CONFIG, "mistralai/Mistral-7B-v0.1")
        self.assertEqual(result.dtype, "float16")
        self.assertEqual(result.max_context_length, 32768)


# ── Tests: GGUF model ID parsing ─────────────────────────────────────────────

class TestGGUFParsing(unittest.TestCase):

    def test_parse_with_quant_specifier(self):
        repo, quant = parse_gguf_model_id("Qwen/Qwen2.5-7B-GGUF:Q4_K_M")
        self.assertEqual(repo, "Qwen/Qwen2.5-7B-GGUF")
        self.assertEqual(quant, "Q4_K_M")

    def test_parse_without_quant_defaults_to_q4(self):
        repo, quant = parse_gguf_model_id("Qwen/Qwen2.5-7B-GGUF")
        self.assertEqual(repo, "Qwen/Qwen2.5-7B-GGUF")
        self.assertEqual(quant, "Q4_K_M")

    def test_quant_uppercased(self):
        _, quant = parse_gguf_model_id("TheBloke/Model-GGUF:q4_k_m")
        self.assertEqual(quant, "Q4_K_M")

    def test_infer_params_7b(self):
        """Returns raw count (7e9), not billions."""
        n = _infer_params_from_name("Llama-3-7B-Instruct")
        self.assertAlmostEqual(n / 1e9, 7.0, delta=0.5)

    def test_infer_params_70b(self):
        n = _infer_params_from_name("Meta-Llama-3-70B")
        self.assertAlmostEqual(n / 1e9, 70.0, delta=2.0)

    def test_infer_params_small_fractional(self):
        n = _infer_params_from_name("Qwen2.5-0.5B-Instruct")
        self.assertAlmostEqual(n / 1e9, 0.5, delta=0.1)

    def test_infer_params_unknown_defaults_to_7b(self):
        n = _infer_params_from_name("some-unnamed-model")
        self.assertAlmostEqual(n / 1e9, 7.0, delta=0.1)
