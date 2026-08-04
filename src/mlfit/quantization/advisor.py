import logging
import re
from typing import Optional

from mlfit.core.models import AlternativeModel
from mlfit.core.memory import estimate_gguf_size_gb, estimate_weights_gb
from mlfit.quantization.quality import get_quality_profile, supported_backends

logger = logging.getLogger(__name__)

_GGUF_QUANT_HIERARCHY = ["Q8_0", "Q6_K", "Q5_K_M", "Q4_K_M", "Q3_K_M", "Q2_K"]
_MAX_ALTERNATIVES = 8


def find_quantized_alternatives(
    model_id: str,
    hardware,
) -> list:
    """
    Find quantized model variants that fit in the available hardware memory.

    Searches for GGUF, AWQ, and GPTQ variants of the model, and checks
    whether CPU offload is feasible. Results are sorted by size ascending
    so the smallest (most likely to fit) appear first.

    Args:
        model_id: HuggingFace model ID, e.g. "meta-llama/Llama-3-70B".
        hardware: HardwareProfile with total_vram_gb, ram_gb, and gpu_backend.

    Returns:
        List of AlternativeModel instances sorted by size, capped at 8 entries.
    """
    available_gb = hardware.total_vram_gb if hardware.gpu_backend != "none" else hardware.ram_gb * 0.6
    param_b = _infer_param_billions(model_id.split("/")[-1])

    logger.info(
        "Searching alternatives for %s (%.0fB params, %.1f GB available)",
        model_id, param_b, available_gb,
    )

    alternatives: list[AlternativeModel] = []
    alternatives.extend(_find_gguf_alternatives(model_id, param_b, available_gb))

    if hardware.gpu_backend in ("cuda", "rocm"):
        awq = _find_awq_alternative(model_id, param_b, available_gb)
        if awq:
            alternatives.append(awq)

        gptq = _find_gptq_alternative(model_id, param_b, available_gb)
        if gptq:
            alternatives.append(gptq)

    cpu_offload = _check_cpu_offload(model_id, param_b, hardware)
    if cpu_offload:
        alternatives.append(cpu_offload)

    alternatives.sort(key=lambda a: a.size_gb)

    logger.info("Found %d alternatives for %s", len(alternatives), model_id)
    return alternatives[:_MAX_ALTERNATIVES]


def _find_gguf_alternatives(
    model_id: str,
    param_b: float,
    available_gb: float,
) -> list:
    """
    Build AlternativeModel entries for each GGUF quant level that fits.

    Walks the quant hierarchy from highest quality (Q8_0) to lowest (Q2_K).
    For each quant that fits in the available memory, resolves the best
    known repo ID via hf_search.

    Args:
        model_id: Original HuggingFace model ID.
        param_b: Estimated parameter count in billions.
        available_gb: Available VRAM or RAM in GB.

    Returns:
        List of AlternativeModel for fitting GGUF variants.
    """
    from mlfit.quantization.hf_search import find_gguf_repos

    results: list[AlternativeModel] = []
    gguf_repos = find_gguf_repos(model_id)
    best_repo = gguf_repos[0] if gguf_repos else model_id

    for quant in _GGUF_QUANT_HIERARCHY:
        size_gb = estimate_gguf_size_gb(quant, param_b)
        if size_gb > available_gb * 0.92:
            logger.debug("GGUF %s skipped: %.1f GB > %.1f GB available", quant, size_gb, available_gb)
            continue

        profile = get_quality_profile(quant)
        alt_id = f"{best_repo}:{quant}" if ":" not in best_repo else best_repo

        results.append(AlternativeModel(
            model_id=alt_id,
            size_gb=size_gb,
            backend=supported_backends(quant),
            quality_loss=f"{profile.label} — {profile.detail}",
            quant_type=quant,
            notes=f"{param_b:.0f}B parameters",
        ))
        logger.debug("GGUF alternative: %s %.1f GB", alt_id, size_gb)

    results.sort(key=lambda a: a.size_gb)
    return results


def _find_awq_alternative(
    model_id: str,
    param_b: float,
    available_gb: float,
) -> Optional[AlternativeModel]:
    """
    Return an AWQ AlternativeModel if a variant exists and fits in memory.

    AWQ (Activation-aware Weight Quantization) is near-lossless int4
    compression optimised for GPU inference on vLLM and TGI.

    Args:
        model_id: Original HuggingFace model ID.
        param_b: Estimated parameter count in billions.
        available_gb: Available VRAM in GB.

    Returns:
        AlternativeModel for the AWQ variant, or None if it does not fit.
    """
    from mlfit.quantization.hf_search import find_awq_repo

    size_gb = estimate_weights_gb(param_b * 1e9, "int4") * 1.15
    if size_gb > available_gb * 0.92:
        logger.debug("AWQ skipped: %.1f GB > %.1f GB available", size_gb, available_gb)
        return None

    repo_id = find_awq_repo(model_id)
    if not repo_id:
        return None

    profile = get_quality_profile("AWQ")
    return AlternativeModel(
        model_id=repo_id,
        size_gb=size_gb,
        backend=supported_backends("AWQ"),
        quality_loss=f"{profile.label} — {profile.detail}",
        quant_type="AWQ",
        notes="Requires CUDA GPU",
    )


def _find_gptq_alternative(
    model_id: str,
    param_b: float,
    available_gb: float,
) -> Optional[AlternativeModel]:
    """
    Return a GPTQ AlternativeModel if a variant exists and fits in memory.

    GPTQ is a post-training quantization method producing int4 weights
    suited for GPU inference on vLLM, TGI, and HuggingFace Transformers.

    Args:
        model_id: Original HuggingFace model ID.
        param_b: Estimated parameter count in billions.
        available_gb: Available VRAM in GB.

    Returns:
        AlternativeModel for the GPTQ variant, or None if it does not fit.
    """
    from mlfit.quantization.hf_search import find_gptq_repo

    size_gb = estimate_weights_gb(param_b * 1e9, "int4") * 1.20
    if size_gb > available_gb * 0.92:
        logger.debug("GPTQ skipped: %.1f GB > %.1f GB available", size_gb, available_gb)
        return None

    repo_id = find_gptq_repo(model_id)
    if not repo_id:
        return None

    profile = get_quality_profile("GPTQ")
    return AlternativeModel(
        model_id=repo_id,
        size_gb=size_gb,
        backend=supported_backends("GPTQ"),
        quality_loss=f"{profile.label} — {profile.detail}",
        quant_type="GPTQ",
        notes="Requires CUDA GPU",
    )


def _check_cpu_offload(
    model_id: str,
    param_b: float,
    hardware,
) -> Optional[AlternativeModel]:
    """
    Check whether the full-precision model can run via CPU+GPU offload.

    When a model is too large for VRAM alone but fits across VRAM + 60% of
    system RAM, llama.cpp can split transformer layers between GPU and CPU.
    This is much slower than full-GPU inference (~0.5× speed) but avoids
    the need to quantize.

    Args:
        model_id: Original HuggingFace model ID.
        param_b: Estimated parameter count in billions.
        hardware: HardwareProfile with total_vram_gb, ram_gb, gpu_backend.

    Returns:
        AlternativeModel describing the offload config, or None if infeasible.
    """
    dtype = "bfloat16"
    model_size_gb = estimate_weights_gb(param_b * 1e9, dtype)
    combined_gb = hardware.total_vram_gb + hardware.ram_gb * 0.60

    if model_size_gb > combined_gb:
        logger.debug(
            "CPU offload infeasible: model=%.1f GB, combined=%.1f GB",
            model_size_gb, combined_gb,
        )
        return None

    name = model_id.split("/")[-1]
    logger.info("CPU offload feasible for %s (%.1f GB model, %.1f GB combined)", name, model_size_gb, combined_gb)

    return AlternativeModel(
        model_id=model_id,
        size_gb=model_size_gb,
        backend="llama.cpp (CPU+GPU offload)",
        quality_loss="none — full precision, ~50% slower than GPU-only",
        quant_type=None,
        notes=(
            f"Split across {hardware.total_vram_gb:.0f} GB VRAM + RAM. "
            "Use llama.cpp with -ngl set to how many layers fit in VRAM."
        ),
    )


def _infer_param_billions(name: str) -> float:
    """
    Guess parameter count in billions from a model name string.

    Scans for patterns like "70B", "7b", "0.5B", "13b" in the name.
    Falls back to 7.0 (a safe default for most mid-size models) when
    no match is found.

    Args:
        name: Model name string, e.g. "Meta-Llama-3-70B-Instruct".

    Returns:
        Parameter count in billions as a float.
    """
    match = re.search(r'(\d+(?:\.\d+)?)\s*[Bb](?:\b|-)', name)
    if match:
        return float(match.group(1))
    logger.debug("Could not infer param count from '%s', defaulting to 7B", name)
    return 7.0
