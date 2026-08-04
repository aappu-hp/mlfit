from mlfit.strategies.base import BaseStrategy, BackendConfig, FeasibilityScore

STRATEGY_REGISTRY: dict = {}   # name → class


def register_strategy(name: str):
    """Decorator to register a backend strategy."""
    def decorator(cls):
        STRATEGY_REGISTRY[name] = cls
        return cls
    return decorator


def _load_all_strategies():
    """Import all strategy modules so they self-register."""
    from mlfit.strategies import vllm, tgi, ollama, llamacpp, onnxruntime, sklearn_strategy  # noqa: F401


__all__ = [
    "BaseStrategy", "BackendConfig", "FeasibilityScore",
    "STRATEGY_REGISTRY", "register_strategy", "_load_all_strategies",
]
