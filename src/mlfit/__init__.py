from mlfit.core.advisor import Advisor
from mlfit.core.models import (
    HardwareProfile,
    ModelProfile,
    AlternativeModel,
    RecommendResult,
    ProfilingResult,
    BenchmarkPoint,
)
from mlfit.strategies.base import BackendConfig, FeasibilityScore
from mlfit.detectors import detect_hardware
from mlfit._version import __version__


def recommend(model_id: str, *, backend: str = "auto", gpuid: str = "all") -> RecommendResult:
    """One-shot recommendation without instantiating Advisor directly."""
    return Advisor(model_id, gpuid=gpuid).recommend(backend=backend)


def profile(
    model_id: str,
    *,
    backend: str = "auto",
    concurrency=None,
    gpuid: str = "all",
) -> ProfilingResult:
    """One-shot profiling without instantiating Advisor directly."""
    return Advisor(model_id, gpuid=gpuid).profile(backend=backend, concurrency=concurrency)


__all__ = [
    "Advisor",
    "recommend",
    "profile",
    "HardwareProfile",
    "ModelProfile",
    "AlternativeModel",
    "RecommendResult",
    "ProfilingResult",
    "BenchmarkPoint",
    "BackendConfig",
    "FeasibilityScore",
    "detect_hardware",
    "__version__",
]
