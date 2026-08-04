from mlfit.core.models import ModelProfile
from mlfit.core.memory import estimate_gguf_size_gb, BYTES_PER_PARAM


def parse_gguf_model_id(model_id: str) -> tuple:
    """
    Parse a GGUF model ID with quant specifier.
    "Qwen/Qwen2.5-7B-GGUF:Q4_K_M" → ("Qwen/Qwen2.5-7B-GGUF", "Q4_K_M")
    "Org/Model-GGUF" → ("Org/Model-GGUF", "Q4_K_M")
    """
    if ":" in model_id:
        repo, quant = model_id.rsplit(":", 1)
        return repo, quant.upper()
    return model_id, "Q4_K_M"


def analyze_hf_gguf(model_id: str) -> ModelProfile:
    """
    Analyze a GGUF model hosted on HuggingFace.
    Downloads a README or the config (not weights) to get metadata.

    model_id format: "Repo/Name:Q4_K_M" or "Repo/Name"
    """
    repo_id, quant_type = parse_gguf_model_id(model_id)
    metadata = _fetch_hf_gguf_metadata(repo_id, quant_type)

    num_params = metadata.get("num_parameters", _infer_params_from_name(repo_id))
    param_b = num_params / 1e9

    size_gb = estimate_gguf_size_gb(quant_type, param_b)
    bpp = BYTES_PER_PARAM.get(quant_type, 0.5)
    dtype_equiv = "int4" if bpp <= 0.6 else "int8" if bpp <= 1.1 else "float16"

    profile = ModelProfile(
        model_id=model_id,
        model_type="gguf",
        num_parameters=num_params,
        dtype=dtype_equiv,
        architecture=metadata.get("architecture", "Unknown"),
        max_context_length=metadata.get("context_length", 4096),
        num_hidden_layers=metadata.get("num_hidden_layers", 32),
        num_attention_heads=metadata.get("num_attention_heads", 32),
        num_key_value_heads=metadata.get("num_key_value_heads", 32),
        hidden_size=metadata.get("hidden_size", 4096),
        quant_type=quant_type,
    )
    profile.estimated_size_gb = size_gb
    return profile


def analyze_gguf_file(path: str) -> ModelProfile:
    """Analyze a local .gguf file by reading its header metadata."""
    try:
        from gguf import GGUFReader
        reader = GGUFReader(path)
        fields = {}
        for key, field in reader.fields.items():
            try:
                if hasattr(field, 'parts') and field.parts:
                    fields[key] = field.parts[-1].tolist()
                    if isinstance(fields[key], list) and len(fields[key]) == 1:
                        fields[key] = fields[key][0]
            except Exception:
                pass

        num_params = fields.get("general.parameter_count", 7e9)
        quant_type = str(fields.get("general.quantization_version", "Q4_K_M"))
        ctx = fields.get("llama.context_length",
                fields.get("phi2.context_length",
                fields.get("general.context_length", 4096)))
        arch = str(fields.get("general.architecture", "llama"))

        size_gb = estimate_gguf_size_gb(quant_type, num_params / 1e9)

        profile = ModelProfile(
            model_id=path,
            model_type="gguf",
            num_parameters=float(num_params),
            dtype="int4",
            architecture=arch,
            max_context_length=int(ctx),
            num_hidden_layers=int(fields.get("llama.block_count", 32)),
            num_attention_heads=int(fields.get("llama.attention.head_count", 32)),
            num_key_value_heads=int(fields.get("llama.attention.head_count_kv", 32)),
            hidden_size=int(fields.get("llama.embedding_length", 4096)),
            quant_type=quant_type,
        )
        profile.estimated_size_gb = size_gb
        return profile

    except ImportError:
        return _estimate_from_filename(path)


def _fetch_hf_gguf_metadata(repo_id: str, quant_type: str) -> dict:
    """
    Fetch GGUF metadata from HuggingFace without downloading the full model.
    Tries to read config.json or README for architecture info.
    """
    metadata = {}
    try:
        from huggingface_hub import hf_hub_download
        import json

        try:
            config_path = hf_hub_download(repo_id, "config.json")
            with open(config_path) as f:
                config = json.load(f)

            metadata["num_parameters"] = _estimate_params_from_config_simple(config)
            metadata["architecture"] = (config.get("architectures", ["Unknown"])[0]
                                        if config.get("architectures") else "Unknown")
            metadata["context_length"] = config.get("max_position_embeddings", 4096)
            metadata["num_hidden_layers"] = config.get("num_hidden_layers", 32)
            metadata["num_attention_heads"] = config.get("num_attention_heads", 32)
            metadata["num_key_value_heads"] = config.get("num_key_value_heads",
                                                          config.get("num_attention_heads", 32))
            metadata["hidden_size"] = config.get("hidden_size", 4096)
            return metadata
        except Exception:
            pass

    except ImportError:
        pass

    metadata["num_parameters"] = _infer_params_from_name(repo_id)
    return metadata


def _infer_params_from_name(model_name: str) -> float:
    """
    Guess raw parameter count from a model name string.

    Returns the absolute count (e.g. 7e9), not billions.
    Callers that need billions must divide by 1e9.

    Examples: "Llama-3-8B" → 8e9,  "Qwen2.5-0.5B" → 0.5e9,  "70B" → 70e9
    """
    import re
    name = model_name.lower()
    match = re.search(r'(\d+(?:\.\d+)?)\s*b(?:\b|-)', name)
    if match:
        return float(match.group(1)) * 1e9
    return 7e9


def _estimate_params_from_config_simple(config: dict) -> float:
    """Simplified parameter estimation from HF config."""
    L = config.get("num_hidden_layers", 32)
    H = config.get("hidden_size", 4096)
    I = config.get("intermediate_size", H * 4)
    V = config.get("vocab_size", 32000)
    return float(V * H + L * (4 * H * H + 2 * H * I))


def _estimate_from_filename(path: str) -> ModelProfile:
    """Last-resort: estimate model info purely from filename."""
    import os
    name = os.path.basename(path)
    num_params = _infer_params_from_name(name)
    size_gb = os.path.getsize(path) / 1e9 if os.path.exists(path) else 4.1

    profile = ModelProfile(
        model_id=path,
        model_type="gguf",
        num_parameters=num_params,
        dtype="int4",
        architecture="Unknown",
        max_context_length=4096,
        num_hidden_layers=32,
        num_attention_heads=32,
        num_key_value_heads=32,
        hidden_size=4096,
        quant_type="Q4_K_M",
    )
    profile.estimated_size_gb = size_gb
    return profile
