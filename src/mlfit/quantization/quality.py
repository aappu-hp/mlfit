from dataclasses import dataclass


@dataclass(frozen=True)
class QualityProfile:
    """
    Describes the quality trade-off for a specific quantization method.

    Attributes:
        label: Short human-readable tier: "minimal", "low", "medium", or "high".
        detail: One-line description of the actual perplexity or quality impact.
        backend: Comma-separated backends that natively support this format.
    """

    label: str
    detail: str
    backend: str


_PROFILES: dict[str, QualityProfile] = {
    "Q8_0":   QualityProfile("minimal", "~0.1% perplexity vs F16 — nearly lossless",   "llama.cpp, Ollama"),
    "Q6_K":   QualityProfile("minimal", "~0.2% perplexity — excellent quality",         "llama.cpp, Ollama"),
    "Q5_K_M": QualityProfile("low",     "~0.5% perplexity — very good quality",         "llama.cpp, Ollama"),
    "Q5_K_S": QualityProfile("low",     "~0.6% perplexity — very good quality",         "llama.cpp, Ollama"),
    "Q4_K_M": QualityProfile("low",     "~1-2% perplexity — recommended default",       "llama.cpp, Ollama"),
    "Q4_K_S": QualityProfile("low",     "~1-3% perplexity — slightly smaller than K_M", "llama.cpp, Ollama"),
    "Q3_K_M": QualityProfile("medium",  "~3-5% perplexity — noticeable quality loss",   "llama.cpp, Ollama"),
    "Q3_K_S": QualityProfile("medium",  "~4-6% perplexity — noticeable quality loss",   "llama.cpp, Ollama"),
    "Q2_K":   QualityProfile("high",    "significant quality loss — last resort only",   "llama.cpp, Ollama"),
    "AWQ":    QualityProfile("minimal", "near-lossless, activation-aware quantization",  "vLLM, TGI"),
    "GPTQ":   QualityProfile("low",     "good quality for GPU inference",                "vLLM, TGI, HF Transformers"),
    "fp8":    QualityProfile("minimal", "near-lossless, requires H100 or newer",         "vLLM"),
}

_FALLBACK = QualityProfile("low", "quality loss depends on model and task", "varies")


def get_quality_profile(quant_type: str) -> QualityProfile:
    """
    Return the QualityProfile for a given quantization type.

    Falls back to a generic low-quality profile when the quant type is
    not in the known table.

    Args:
        quant_type: Quantization identifier, e.g. "Q4_K_M", "AWQ", "GPTQ".

    Returns:
        QualityProfile with label, detail, and backend fields.
    """
    return _PROFILES.get(quant_type, _FALLBACK)


def quality_label(quant_type: str) -> str:
    """
    Return just the quality tier label for display in summary tables.

    Args:
        quant_type: Quantization identifier.

    Returns:
        One of "minimal", "low", "medium", "high", or "low" for unknowns.
    """
    return get_quality_profile(quant_type).label


def supported_backends(quant_type: str) -> str:
    """
    Return the comma-separated list of backends that support this quant type.

    Args:
        quant_type: Quantization identifier.

    Returns:
        Human-readable backend string, e.g. "llama.cpp, Ollama".
    """
    return get_quality_profile(quant_type).backend
