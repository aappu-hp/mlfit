from dataclasses import dataclass

from mlfit.analyzers.hf_analyzer import estimate_params_from_config
from mlfit.tui.fit import FitLevel, classify_fit


@dataclass
class ModelRow:
    """
    Display metadata for one model row in the browser table.

    All fields are derived from the locally cached config, so building a row
    involves no network calls.
    """

    model_id: str      # "Qwen/Qwen2.5-7B"
    name: str          # "Qwen2.5-7B"
    provider: str      # "Qwen"
    params_b: float    # billions of parameters; 0.0 if unknown
    context: int       # context window in tokens; 0 if unknown
    fit: FitLevel


def _params_billions(config: dict) -> float:
    """
    Return parameter count in billions.

    Prefers the exact safetensors total cached under ``_mlfit_params_total``,
    falling back to the architecture-based estimate. Returns 0.0 if neither is
    available.
    """
    exact = config.get("_mlfit_params_total")
    if exact:
        return exact / 1e9
    try:
        return estimate_params_from_config(config) / 1e9
    except Exception:
        return 0.0


def _context_length(config: dict) -> int:
    """Return the model's context window from the config, or 0 if unknown."""
    value = config.get(
        "max_position_embeddings",
        config.get("n_positions", config.get("seq_length", 0)),
    )
    try:
        return int(value or 0)
    except (TypeError, ValueError):
        return 0


def build_row(model_id: str, config: dict | None, vram_gb: float) -> ModelRow:
    """
    Build a display row for a model from its cached config.

    Args:
        model_id: HuggingFace model ID ("org/name").
        config: Cached config dict, or None if the config is unavailable.
        vram_gb: Available VRAM (or unified memory) in GB, for the fit estimate.

    Returns:
        A ModelRow with provider/name split from the id, params and context from
        the config, and a FitLevel from the estimated memory footprint.
    """
    org, _, name = model_id.partition("/")
    if not name:
        name, org = org, ""

    params_b = _params_billions(config) if config else 0.0
    context = _context_length(config) if config else 0

    if params_b > 0:
        estimated_gb = params_b * 2 * 1.2  # fp16 weights + 20% overhead
        fit = classify_fit(estimated_gb, vram_gb)
    else:
        fit = FitLevel.UNKNOWN

    return ModelRow(
        model_id=model_id,
        name=name or model_id,
        provider=org,
        params_b=params_b,
        context=context,
        fit=fit,
    )


def fmt_params(params_b: float) -> str:
    """Format a parameter count in billions as a compact string, e.g. '7.6B'."""
    if params_b <= 0:
        return "—"
    if params_b < 1:
        return f"{params_b * 1000:.0f}M"
    return f"{params_b:.1f}B"


def fmt_ctx(context: int) -> str:
    """Format a context window as a compact string, e.g. '32k', '131k'."""
    if context <= 0:
        return "—"
    if context < 1000:
        return str(context)
    return f"{context // 1000}k"
