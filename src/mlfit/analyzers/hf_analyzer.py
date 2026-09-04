import json
import logging

from mlfit.core.models import ModelProfile
from mlfit.core.memory import estimate_weights_gb
from mlfit.exceptions import ModelNotFoundError

logger = logging.getLogger(__name__)


def _hf_hub_download(repo_id: str, filename: str) -> str:
    """
    Thin wrapper around huggingface_hub.hf_hub_download.

    Exists as a module-level name so tests can patch it:
        patch("mlfit.analyzers.hf_analyzer._hf_hub_download", ...)

    Without this wrapper, hf_hub_download is only a local name inside
    _download_config and cannot be intercepted by unittest.mock.patch.
    """
    from huggingface_hub import hf_hub_download
    return hf_hub_download(repo_id=repo_id, filename=filename)


def analyze_hf_model(model_id: str) -> ModelProfile:
    """
    Analyze a HuggingFace model without downloading weights.
    Downloads only config.json (~5–10KB).

    Args:
        model_id: HuggingFace model ID, e.g. "meta-llama/Llama-3-8B"

    Returns:
        ModelProfile with architecture info and size estimates
    """
    logger.debug("Downloading config.json for %s", model_id)
    config = _download_config(model_id)
    logger.debug("Config downloaded successfully for %s", model_id)

    num_params = _estimate_params_from_config(config)
    dtype = _resolve_dtype(config)
    max_ctx = config.get("max_position_embeddings",
                config.get("n_positions",
                config.get("seq_length", 4096)))

    num_heads = config.get("num_attention_heads", 32)
    num_kv_heads = config.get("num_key_value_heads", num_heads)

    profile = ModelProfile(
        model_id=model_id,
        model_type="llm",
        num_parameters=num_params,
        dtype=dtype,
        architecture=config.get("architectures", ["Unknown"])[0]
                     if isinstance(config.get("architectures"), list)
                     else config.get("model_type", "Unknown"),
        max_context_length=int(max_ctx),
        num_hidden_layers=config.get("num_hidden_layers",
                          config.get("n_layer", 32)),
        num_attention_heads=num_heads,
        num_key_value_heads=num_kv_heads,
        hidden_size=config.get("hidden_size",
                    config.get("n_embd", 4096)),
        quant_type=None,
    )
    profile.estimated_size_gb = estimate_weights_gb(num_params, dtype)
    return profile


def _download_config(model_id: str) -> dict:
    """Download config.json from HuggingFace Hub. No weights downloaded."""
    from huggingface_hub.utils import RepositoryNotFoundError, EntryNotFoundError

    try:
        config_path = _hf_hub_download(repo_id=model_id, filename="config.json")
        with open(config_path) as f:
            return json.load(f)
    except (RepositoryNotFoundError, EntryNotFoundError):
        raise ModelNotFoundError(model_id)


def _estimate_params_from_config(config: dict) -> float:
    """
    Estimate parameter count from transformer architecture config.

    Uses the standard LLaMA/GPT-style formula:
      embedding_params + num_layers × (attention_params + FFN_params) + output_params

    This is approximate but accurate to within ~5% for most transformers.
    """
    num_layers = config.get("num_hidden_layers", config.get("n_layer", 32))
    hidden_size = config.get("hidden_size", config.get("n_embd", 4096))
    intermediate_size = config.get("intermediate_size", hidden_size * 4)
    vocab_size = config.get("vocab_size", 32000)
    num_heads = config.get("num_attention_heads", 32)
    num_kv_heads = config.get("num_key_value_heads", num_heads)

    embedding = vocab_size * hidden_size

    head_dim = hidden_size // max(num_heads, 1)
    q_size = hidden_size * hidden_size
    kv_size = num_kv_heads * head_dim * hidden_size
    attention = q_size + 2 * kv_size + hidden_size * hidden_size

    ffn_type = config.get("hidden_act", "")
    if any(x in ffn_type for x in ["silu", "swiglu", "gelu_new"]):
        ffn = hidden_size * intermediate_size + hidden_size * intermediate_size + intermediate_size * hidden_size
    else:
        ffn = hidden_size * intermediate_size + intermediate_size * hidden_size

    layer_norms = 2 * hidden_size * num_layers
    per_layer = attention + ffn
    total = embedding + num_layers * per_layer + layer_norms

    return float(total)


def _resolve_dtype(config: dict) -> str:
    """
    Resolve the model's native dtype from config.
    Defaults to bfloat16 for modern models, float16 for older ones.
    """
    dtype = config.get("torch_dtype", "")
    if dtype in ("float32", "float16", "bfloat16"):
        return dtype

    architectures = config.get("architectures", [])
    arch_str = " ".join(architectures).lower() if architectures else ""

    modern_archs = ["llama", "mistral", "qwen", "gemma", "phi", "falcon"]
    if any(a in arch_str for a in modern_archs):
        return "bfloat16"

    return "float16"
