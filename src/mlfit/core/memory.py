BYTES_PER_PARAM = {
    "float32": 4.0,
    "float16": 2.0,
    "bfloat16": 2.0,
    "int8": 1.0,
    "int4": 0.5,
    "Q2_K":   0.325,
    "Q4_K_M": 0.5625,
    "Q5_K_M": 0.6875,
    "Q8_0":   1.0625,
    "F16":    2.0,
}

GGUF_SIZES = {
    ("Q2_K",   7):  2.8,  ("Q4_K_M",  7):  4.1,  ("Q5_K_M",  7):  5.1,  ("Q8_0",  7):  7.7,
    ("Q2_K",  13):  5.2,  ("Q4_K_M", 13):  7.6,  ("Q5_K_M", 13):  9.3,  ("Q8_0", 13): 14.0,
    ("Q2_K",  70): 26.0,  ("Q4_K_M", 70): 41.0,  ("Q5_K_M", 70): 48.0,  ("Q8_0", 70): 74.0,
}


def estimate_weights_gb(num_params: float, dtype: str) -> float:
    """Quick estimate: weight bytes only."""
    bpp = BYTES_PER_PARAM.get(dtype, 2.0)
    return (num_params * bpp) / 1e9


def estimate_vram_gb(model, max_seq_len: int = 4096, max_batch: int = 16) -> float:
    """
    Full VRAM estimate including weights, KV cache, activations, and overhead.

    Args:
        model: ModelProfile instance
        max_seq_len: context length to use for KV cache calculation
        max_batch: number of concurrent sequences (for KV cache)

    Returns:
        Estimated total VRAM in GB
    """
    dtype_key = model.quant_type if model.quant_type else model.dtype
    bpp = BYTES_PER_PARAM.get(dtype_key, 2.0)

    weights_gb = (model.num_parameters * bpp) / 1e9

    if model.num_attention_heads > 0 and model.hidden_size > 0:
        head_dim = model.hidden_size // model.num_attention_heads
        kv_heads = model.num_key_value_heads or model.num_attention_heads
        kv_cache_bytes = (
            2 * model.num_hidden_layers * kv_heads
            * head_dim * max_seq_len * max_batch * bpp
        )
        kv_cache_gb = kv_cache_bytes / 1e9
    else:
        kv_cache_gb = 0.0

    activations_gb = weights_gb * 0.15
    overhead_gb = 0.75

    return weights_gb + kv_cache_gb + activations_gb + overhead_gb


def estimate_gguf_size_gb(quant_type: str, param_billions: float) -> float:
    """
    Estimate GGUF file size in GB given quant type and parameter count.
    Uses lookup table with linear interpolation for intermediate sizes.
    """
    known_sizes = [7, 13, 70]
    closest = min(known_sizes, key=lambda x: abs(x - param_billions))
    key = (quant_type, closest)

    if key in GGUF_SIZES:
        base_size = GGUF_SIZES[key]
        scale = param_billions / closest
        return base_size * scale

    bpp = BYTES_PER_PARAM.get(quant_type, 0.5)
    return (param_billions * 1e9 * bpp) / 1e9


def check_fits(model, hardware) -> tuple:
    """
    Check if model fits in available VRAM/RAM.

    Returns:
        (fits: bool, estimated_vram_gb: float, headroom_gb: float)
    """
    estimated = estimate_vram_gb(model, max_seq_len=4096, max_batch=1)

    if hardware.gpu_backend != "none" and hardware.total_vram_gb > 0:
        available = hardware.total_vram_gb
        fits = estimated <= available * 0.95
        headroom = available - estimated
    else:
        available = hardware.ram_gb * 0.7
        weights_gb = estimate_weights_gb(model.num_parameters,
                                         model.quant_type or model.dtype)
        fits = weights_gb <= available
        headroom = available - weights_gb
        estimated = weights_gb

    return fits, estimated, headroom
