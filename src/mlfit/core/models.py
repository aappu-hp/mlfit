from dataclasses import dataclass, field
from typing import Optional, TYPE_CHECKING

if TYPE_CHECKING:
    from mlfit.strategies.base import BackendConfig


@dataclass
class GPUInfo:
    """Represents a single GPU device and its memory state at detection time."""

    index: int
    name: str
    vram_total_gb: float
    vram_free_gb: float
    compute_capability: tuple  # (major, minor) e.g. (8, 9) = Ada Lovelace
    is_unified_memory: bool    # True for Apple Silicon unified memory architecture


@dataclass
class HardwareProfile:
    """Normalized snapshot of all hardware resources relevant to model serving."""

    gpus: list                 # list[GPUInfo]
    total_vram_gb: float       # sum of free VRAM across selected GPUs
    gpu_backend: str           # "cuda" | "rocm" | "mps" | "none"
    cpu_cores: int
    cpu_threads: int
    ram_gb: float
    has_avx: bool
    has_avx2: bool
    has_avx512: bool
    disk_free_gb: float
    ram_available_gb: float = 0.0   # currently free system RAM
    swap_total_gb: float = 0.0
    swap_used_gb: float = 0.0


@dataclass
class ModelProfile:
    """Architecture and size metadata for a model, derived without downloading weights."""

    model_id: str
    model_type: str                        # "llm" | "gguf" | "onnx" | "sklearn" | "vision"
    num_parameters: float                  # e.g. 8e9
    dtype: str                             # "float16" | "bfloat16" | "float32"
    architecture: str                      # "LlamaForCausalLM" etc.
    max_context_length: int
    num_hidden_layers: int
    num_attention_heads: int
    num_key_value_heads: int               # GQA — may differ from attention_heads
    hidden_size: int
    quant_type: Optional[str] = None       # "Q4_K_M" | "AWQ" | "GPTQ" | None
    estimated_size_gb: float = 0.0         # filled by analyzer

    @property
    def effective_dtype(self) -> str:
        """Return the active dtype: quant_type when set, otherwise dtype."""
        return self.quant_type or self.dtype


@dataclass
class AlternativeModel:
    """A quantized or smaller model that can serve as a drop-in replacement."""

    model_id: str
    size_gb: float
    backend: str
    quality_loss: str                      # "minimal" | "low" | "medium" | "high"
    quant_type: Optional[str] = None
    notes: str = ""


@dataclass
class RecommendResult:
    """
    Full output of a recommendation run: hardware context, model analysis,
    ranked backend configurations, and quantized fallback options.
    """

    hardware: HardwareProfile
    model: ModelProfile
    configs: list                          # list of (FeasibilityScore, BackendConfig, Strategy)
    is_feasible: bool
    quantized_alternatives: list           # list[AlternativeModel]
    estimated_vram_gb: float = 0.0         # total VRAM needed (weights + KV cache + overhead)
    headroom_gb: float = 0.0               # available VRAM minus estimated VRAM

    @property
    def best(self):
        """The highest-ranked BackendConfig, or None if no compatible backends were found."""
        if self.configs:
            return self.configs[0][1]
        return None

    @property
    def alternatives(self):
        """All BackendConfigs after the top-ranked one, in descending feasibility order."""
        return [c[1] for c in self.configs[1:]]

    def to_dict(self) -> dict:
        """
        Serialise the result to a JSON-compatible dictionary.

        Returns:
            Dictionary with mlfit_version, timestamp, hardware, model,
            configs, and quantized_alternatives.
        """
        import datetime
        from mlfit._version import __version__

        hardware = self.hardware
        model = self.model

        return {
            "mlfit_version": __version__,
            "timestamp": datetime.datetime.now(datetime.timezone.utc).isoformat().replace("+00:00", "Z"),
            "hardware": {
                "gpus": [
                    {
                        "index": g.index,
                        "name": g.name,
                        "vram_total_gb": round(g.vram_total_gb, 2),
                        "vram_free_gb": round(g.vram_free_gb, 2),
                        "compute_capability": list(g.compute_capability),
                        "is_unified_memory": g.is_unified_memory,
                    }
                    for g in hardware.gpus
                ],
                "total_vram_gb": round(hardware.total_vram_gb, 2),
                "gpu_backend": hardware.gpu_backend,
                "cpu_cores": hardware.cpu_cores,
                "cpu_threads": hardware.cpu_threads,
                "ram_gb": round(hardware.ram_gb, 1),
                "has_avx": hardware.has_avx,
                "has_avx2": hardware.has_avx2,
                "has_avx512": hardware.has_avx512,
                "disk_free_gb": round(hardware.disk_free_gb, 1),
            },
            "model": {
                "model_id": model.model_id,
                "model_type": model.model_type,
                "num_parameters": int(model.num_parameters),
                "dtype": model.dtype,
                "architecture": model.architecture,
                "max_context_length": model.max_context_length,
                "estimated_size_gb": round(model.estimated_size_gb, 2),
                "quant_type": model.quant_type,
            },
            "is_feasible": self.is_feasible,
            "configs": [
                {
                    "rank": i + 1,
                    "backend": cfg.backend,
                    "feasibility_score": round(score.score, 2),
                    "estimated_tps": round(cfg.estimated_tps, 1),
                    "estimated_vram_gb": round(cfg.estimated_vram_gb, 2),
                    "params": cfg.params,
                    "command": cfg.command,
                }
                for i, (score, cfg, _) in enumerate(self.configs)
            ],
            "quantized_alternatives": [
                {
                    "model_id": alt.model_id,
                    "size_gb": round(alt.size_gb, 2),
                    "backend": alt.backend,
                    "quality_loss": alt.quality_loss,
                    "quant_type": alt.quant_type,
                }
                for alt in self.quantized_alternatives
            ],
        }

    def to_json(self) -> str:
        """
        Serialise the result to a formatted JSON string.

        Returns:
            Indented JSON string representation of to_dict().
        """
        import json
        return json.dumps(self.to_dict(), indent=2)


@dataclass
class BenchmarkPoint:
    """Throughput and latency measurements at a single concurrency level."""

    concurrency: int
    tps: float
    ttft_p50_ms: float
    ttft_p95_ms: float


@dataclass
class ProfilingResult:
    """
    Full output of a dynamic profiling run.

    Unlike RecommendResult (static estimates), every field here is a
    real measurement taken against a live inference server.
    """

    model_id: str
    backend: str
    hardware: "HardwareProfile"
    final_config: "BackendConfig"
    peak_vram_gb: float
    benchmark_points: list            # list[BenchmarkPoint]
    time_elapsed_s: float
    timestamp: str

    def to_dict(self) -> dict:
        """
        Serialise the profiling result to a JSON-compatible dictionary.

        Returns:
            Dictionary containing hardware, config, memory, and benchmark data.
        """
        from mlfit._version import __version__

        hardware = self.hardware
        cfg = self.final_config

        return {
            "mlfit_version": __version__,
            "timestamp": self.timestamp,
            "model_id": self.model_id,
            "backend": self.backend,
            "hardware": {
                "gpus": [
                    {
                        "index": g.index,
                        "name": g.name,
                        "vram_total_gb": round(g.vram_total_gb, 2),
                        "vram_free_gb": round(g.vram_free_gb, 2),
                    }
                    for g in hardware.gpus
                ],
                "gpu_backend": hardware.gpu_backend,
                "ram_gb": round(hardware.ram_gb, 1),
            },
            "final_config": {
                "params": cfg.params,
                "command": cfg.command,
            },
            "memory": {
                "peak_vram_gb": round(self.peak_vram_gb, 2),
            },
            "benchmarks": [
                {
                    "concurrency": bp.concurrency,
                    "tps": round(bp.tps, 1),
                    "ttft_p50_ms": round(bp.ttft_p50_ms, 1),
                    "ttft_p95_ms": round(bp.ttft_p95_ms, 1),
                }
                for bp in self.benchmark_points
            ],
            "time_elapsed_s": round(self.time_elapsed_s, 1),
        }

    def tps_at_concurrency(self, n: int) -> float:
        """Return the measured TPS at a specific concurrency level.

        Args:
            n: Concurrency level that was benchmarked (e.g. 1, 4, 16).

        Raises:
            ValueError: If that concurrency level was not included in the run.
        """
        for bp in self.benchmark_points:
            if bp.concurrency == n:
                return bp.tps
        available = [bp.concurrency for bp in self.benchmark_points]
        raise ValueError(
            f"Concurrency {n} was not benchmarked. Available levels: {available}"
        )

    def to_json(self) -> str:
        """
        Serialise the profiling result to a formatted JSON string.

        Returns:
            Indented JSON string representation of to_dict().
        """
        import json
        return json.dumps(self.to_dict(), indent=2)
