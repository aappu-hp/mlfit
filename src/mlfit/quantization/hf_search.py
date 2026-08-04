import logging
from typing import Optional

logger = logging.getLogger(__name__)

# Publishers tried in priority order when searching for GGUF variants.
# bartowski is the current most actively maintained source;
# TheBloke has the largest legacy catalog.
_GGUF_PUBLISHERS = ["bartowski", "TheBloke"]


def list_models(*args, **kwargs):
    """
    Module-level wrapper around huggingface_hub.list_models.

    Exists as a named module attribute so tests can patch it cleanly:
        patch("mlfit.quantization.hf_search.list_models", ...)
    """
    from huggingface_hub import list_models as _list
    return _list(*args, **kwargs)


def find_gguf_repos(model_id: str) -> list:
    """
    Return candidate GGUF repo IDs for a given model, in priority order.

    Confirmed HuggingFace search results come first because they are known
    to exist. Constructed fallbacks (bartowski, TheBloke, official org) follow
    for use when the search returns nothing.

    Args:
        model_id: HuggingFace model ID, e.g. "meta-llama/Llama-3-8B-Instruct".

    Returns:
        List of repo ID strings, confirmed results first then constructed fallbacks.
        May be empty if nothing is found.
    """
    org, name = _split_model_id(model_id)

    confirmed = _search_hf_for_gguf(name)
    logger.debug("HF search returned %d confirmed GGUF repos for %s", len(confirmed), model_id)

    fallbacks = [f"{publisher}/{name}-GGUF" for publisher in _GGUF_PUBLISHERS]
    fallbacks.append(f"{org}/{name}-GGUF")

    unique_fallbacks = [f for f in fallbacks if f not in confirmed]

    result = confirmed + unique_fallbacks
    logger.info("Found %d GGUF candidate repos for %s", len(result), model_id)
    return result


def find_awq_repo(model_id: str) -> Optional[str]:
    """
    Find an AWQ quantized variant of a model on HuggingFace.

    Tries the official org repo first, then falls back to keyword search.
    AWQ variants require a CUDA GPU and are supported by vLLM and TGI.

    Args:
        model_id: HuggingFace model ID.

    Returns:
        Repo ID string if found, None otherwise.
    """
    org, name = _split_model_id(model_id)

    official_awq = f"{org}/{name}-AWQ"
    logger.debug("Checking official AWQ repo: %s", official_awq)

    try:
        results = list(list_models(search=f"{name} AWQ", limit=5))
        for result in results:
            rid = result.modelId
            if "awq" in rid.lower() and _name_matches(name, rid):
                logger.info("Found AWQ repo via search: %s", rid)
                return rid
    except Exception:
        logger.debug("AWQ HF search failed for %s", model_id, exc_info=True)

    logger.debug("No AWQ repo found for %s, using constructed ID", model_id)
    return official_awq


def find_gptq_repo(model_id: str) -> Optional[str]:
    """
    Find a GPTQ quantized variant of a model on HuggingFace.

    Tries the official org repo first, then checks TheBloke (who has a large
    GPTQ catalog), then falls back to keyword search.

    Args:
        model_id: HuggingFace model ID.

    Returns:
        Repo ID string if found, None otherwise.
    """
    org, name = _split_model_id(model_id)

    candidates_to_try = [
        f"{org}/{name}-GPTQ",
        f"TheBloke/{name}-GPTQ",
    ]

    try:
        results = list(list_models(search=f"{name} GPTQ", limit=8))
        for result in results:
            rid = result.modelId
            if "gptq" in rid.lower() and _name_matches(name, rid):
                logger.info("Found GPTQ repo via search: %s", rid)
                return rid
    except Exception:
        logger.debug("GPTQ HF search failed for %s", model_id, exc_info=True)

    logger.debug("Returning constructed GPTQ ID for %s", model_id)
    return candidates_to_try[0]


def _search_hf_for_gguf(model_name: str) -> list:
    """
    Search HuggingFace for GGUF repos matching a model name.

    Args:
        model_name: The model name portion (without org prefix).

    Returns:
        List of repo ID strings from the search results.
    """
    try:
        results = list(list_models(search=f"{model_name} GGUF", filter="gguf", limit=10))
        matched = [
            r.modelId for r in results
            if _name_matches(model_name, r.modelId)
        ]
        logger.debug(
            "HF GGUF search for '%s' returned %d results, %d matched",
            model_name, len(results), len(matched),
        )
        return matched
    except Exception:
        logger.debug("HF GGUF search failed for '%s'", model_name, exc_info=True)
        return []


def _split_model_id(model_id: str) -> tuple:
    """
    Split a model ID into (org, name) parts.

    Args:
        model_id: HuggingFace model ID, e.g. "meta-llama/Llama-3-8B".

    Returns:
        Tuple of (org, name). If no slash is present, org defaults to "unknown".
    """
    parts = model_id.split("/", 1)
    if len(parts) == 2:
        return parts[0], parts[1]
    return "unknown", parts[0]


def _name_matches(model_name: str, repo_id: str) -> bool:
    """
    Check whether a repo ID is likely a variant of the given model name.

    Uses a case-insensitive substring check on the core model name tokens
    to avoid returning unrelated models from broad keyword searches.

    Args:
        model_name: Base model name, e.g. "Llama-3-8B-Instruct".
        repo_id: Full repo ID to check, e.g. "bartowski/Llama-3-8B-Instruct-GGUF".

    Returns:
        True if the repo is likely a match for the model.
    """
    core = _extract_core_name(model_name).lower()
    return core in repo_id.lower()


def _extract_core_name(model_name: str) -> str:
    """
    Extract the essential identifying tokens from a model name.

    Strips common suffixes (Instruct, Chat, v0.1, etc.) to get the
    canonical name used for matching across repos.

    Args:
        model_name: Full model name, e.g. "Llama-3-8B-Instruct".

    Returns:
        Shortened core name, e.g. "Llama-3-8B".
    """
    import re
    name = re.sub(
        r'[-_](instruct|chat|it|hf|base|v\d+[\.\d]*)$',
        '',
        model_name,
        flags=re.IGNORECASE,
    )
    return name
