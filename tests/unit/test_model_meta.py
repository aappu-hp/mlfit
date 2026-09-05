from unittest.mock import patch

from mlfit.tui.fit import FitLevel
from mlfit.tui.model_meta import ModelRow, build_row, fmt_ctx, fmt_params
from mlfit.analyzers import hf_analyzer


_QWEN_CONFIG = {
    "hidden_size": 3584,
    "num_hidden_layers": 28,
    "intermediate_size": 18944,
    "vocab_size": 152064,
    "num_attention_heads": 28,
    "num_key_value_heads": 4,
    "max_position_embeddings": 32768,
}


def test_build_row_prefers_exact_param_count():
    config = {**_QWEN_CONFIG, "_mlfit_params_total": 7_620_000_000}
    row = build_row("Qwen/Qwen2.5-7B", config, vram_gb=17.0)
    assert row.provider == "Qwen"
    assert row.name == "Qwen2.5-7B"
    assert round(row.params_b, 2) == 7.62
    assert row.context == 32768
    assert row.fit is not FitLevel.UNKNOWN


def test_build_row_falls_back_to_formula():
    row = build_row("Qwen/Qwen2.5-7B", _QWEN_CONFIG, vram_gb=17.0)
    assert row.params_b > 0            # formula produced an estimate
    assert row.context == 32768
    assert row.fit is not FitLevel.UNKNOWN


def test_build_row_unknown_when_no_config():
    row = build_row("some/model", None, vram_gb=17.0)
    assert row.provider == "some"
    assert row.name == "model"
    assert row.params_b == 0.0
    assert row.context == 0
    assert row.fit is FitLevel.UNKNOWN


def test_build_row_handles_id_without_org():
    row = build_row("standalone", {}, vram_gb=17.0)
    assert row.name == "standalone"
    assert row.provider == ""


def test_fmt_params():
    assert fmt_params(7.62) == "7.6B"
    assert fmt_params(0.36) == "360M"
    assert fmt_params(0) == "—"


def test_fmt_ctx():
    assert fmt_ctx(32768) == "32k"
    assert fmt_ctx(131072) == "131k"
    assert fmt_ctx(512) == "512"
    assert fmt_ctx(0) == "—"


def test_fetch_param_count_reads_safetensors_total():
    class _Safe:
        total = 7_620_000_000

    class _Info:
        safetensors = _Safe()

    with patch.object(hf_analyzer, "_hf_model_info", return_value=_Info()):
        assert hf_analyzer.fetch_param_count("Qwen/Qwen2.5-7B") == 7_620_000_000


def test_fetch_param_count_none_without_safetensors():
    class _Info:
        safetensors = None

    with patch.object(hf_analyzer, "_hf_model_info", return_value=_Info()):
        assert hf_analyzer.fetch_param_count("some/gguf-only") is None


def test_fetch_param_count_none_on_error():
    with patch.object(hf_analyzer, "_hf_model_info", side_effect=RuntimeError("offline")):
        assert hf_analyzer.fetch_param_count("x/y") is None
