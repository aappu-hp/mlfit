import importlib.util
import os
import pathlib
import shutil


def _has_module(module_name: str) -> bool:
    """Return True if the given Python module is importable in this environment."""
    try:
        return importlib.util.find_spec(module_name) is not None
    except (ImportError, ValueError):
        return False


def detect_runtimes() -> dict[str, bool]:
    """
    Detect which local model-serving runtimes are available on this system.

    Uses fast, side-effect-free checks only (PATH lookups, module resolution,
    and known install locations) — never spawns subprocesses that could hang.

    Returns:
        Mapping of runtime display name to availability, in display order.
    """
    return {
        "Ollama": bool(shutil.which("ollama")),
        "llama.cpp": bool(
            shutil.which("llama-cli")
            or shutil.which("llama-server")
            or os.getenv("LLAMA_CPP_PATH")
        ),
        "vLLM": _has_module("vllm"),
        "MLX": _has_module("mlx") or _has_module("mlx_lm"),
        "LM Studio": bool(shutil.which("lms"))
        or pathlib.Path("/Applications/LM Studio.app").exists(),
    }
