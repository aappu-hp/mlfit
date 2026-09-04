from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from mlfit.core.models import ModelProfile, HardwareProfile


@dataclass
class FeasibilityScore:
    """
    A 0–1 score representing how well a backend suits a model/hardware combination.

    Attributes:
        score: Normalised score from 0.0 (incompatible) to 1.0 (perfect fit).
        reason: Human-readable explanation of the score.
        warnings: Optional list of non-fatal concern strings shown to the user.
    """

    score: float
    reason: str
    warnings: list = field(default_factory=list)


@dataclass
class BackendConfig:
    """
    A fully resolved serving configuration for a specific backend.

    Attributes:
        backend: Backend name, e.g. "vllm" or "ollama".
        model_id: The HuggingFace or local model identifier.
        params: Backend-specific key/value parameters.
        command: Exact CLI command the user can run to start the server.
        estimated_vram_gb: Predicted VRAM consumption in GB.
        estimated_tps: Predicted tokens-per-second throughput.
        server_url: Base URL where the server listens (used by the profiler).
        server_command: Daemon command to start the server, if different from command.
        health_path: Path the profiler polls to confirm the server is ready.
    """

    backend: str
    model_id: str
    params: dict
    command: str
    estimated_vram_gb: float
    estimated_tps: float
    server_url: str = "http://localhost:8000"
    server_command: str = ""
    health_path: str = "/health"


class BaseStrategy(ABC):
    """Abstract base for all backend strategies."""

    @abstractmethod
    def is_compatible(self, model: "ModelProfile", hardware: "HardwareProfile") -> bool:
        """
        Determine whether this backend can run the model on the given hardware.

        Args:
            model: ModelProfile describing architecture and size.
            hardware: HardwareProfile describing available resources.

        Returns:
            True if the backend is usable; False otherwise.
        """
        ...

    @abstractmethod
    def estimate_feasibility(
        self, model: "ModelProfile", hardware: "HardwareProfile"
    ) -> FeasibilityScore:
        """
        Score how well this backend suits the model/hardware combination.

        Args:
            model: ModelProfile describing architecture and size.
            hardware: HardwareProfile describing available resources.

        Returns:
            FeasibilityScore with a 0–1 score and a human-readable reason.
        """
        ...

    @abstractmethod
    def generate_config(
        self, model: "ModelProfile", hardware: "HardwareProfile"
    ) -> BackendConfig:
        """
        Produce the optimal BackendConfig for this model/hardware pair.

        Args:
            model: ModelProfile describing architecture and size.
            hardware: HardwareProfile describing available resources.

        Returns:
            BackendConfig with resolved parameters and a ready-to-run command.
        """
        ...

    def format_key_settings(self, params: dict) -> str:
        """
        Format the most important parameters as a compact human-readable string.

        Override in concrete strategies to highlight backend-specific params.
        The default implementation joins all key=value pairs.

        Args:
            params: The BackendConfig.params dict for this configuration.

        Returns:
            Compact string for display in the recommendations table.
        """
        return ", ".join(f"{k}={v}" for k, v in params.items()) or "defaults"

    def _gpu_name(self, hardware: "HardwareProfile") -> str:
        """Return the first GPU's name in lowercase, or empty string if CPU-only.

        Args:
            hardware: HardwareProfile with a gpus list.

        Returns:
            Lowercase GPU name string, or "" when no GPU is present.
        """
        return hardware.gpus[0].name.lower() if hardware.gpus else ""

    def _param_scale(self, model: "ModelProfile") -> float:
        """Return a 0–1 TPS scale factor based on parameter count relative to 7B.

        A 7B model scores 1.0; larger models score proportionally lower.

        Args:
            model: ModelProfile with num_parameters set.

        Returns:
            Float in (0, 1] used to scale base TPS estimates.
        """
        params_b = model.num_parameters / 1e9
        return min(7.0 / max(params_b, 0.5), 1.0)

    def generate_config_candidates(
        self, model: "ModelProfile", hardware: "HardwareProfile"
    ) -> list:
        """
        Return a list of BackendConfigs ordered from conservative to aggressive.

        The profiler iterates through candidates until one successfully loads.
        Override in strategies that support memory-utilization tuning (e.g. vLLM).
        The default returns a single config from generate_config().

        Args:
            model: ModelProfile describing architecture and size.
            hardware: HardwareProfile describing available resources.

        Returns:
            Non-empty list of BackendConfig instances, least aggressive first.
        """
        return [self.generate_config(model, hardware)]
