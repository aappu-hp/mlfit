from mlfit.strategies.base import BaseStrategy, BackendConfig, FeasibilityScore
from mlfit.strategies import register_strategy

OLLAMA_MODEL_MAP = {
    "meta-llama/Llama-3-8B-Instruct": "llama3:8b-instruct",
    "meta-llama/Llama-3-70B-Instruct": "llama3:70b-instruct",
    "meta-llama/Meta-Llama-3-8B-Instruct": "llama3:8b-instruct",
    "meta-llama/Meta-Llama-3.1-8B-Instruct": "llama3.1:8b-instruct",
    "meta-llama/Llama-3.2-3B-Instruct": "llama3.2:3b-instruct",
    "mistralai/Mistral-7B-Instruct-v0.3": "mistral:7b-instruct",
    "mistralai/Mistral-7B-v0.1": "mistral:7b",
    "Qwen/Qwen2.5-7B-Instruct": "qwen2.5:7b-instruct",
    "Qwen/Qwen2.5-14B-Instruct": "qwen2.5:14b-instruct",
    "Qwen/Qwen2.5-72B-Instruct": "qwen2.5:72b-instruct",
    "google/gemma-2-9b-it": "gemma2:9b-instruct",
    "microsoft/Phi-3-mini-4k-instruct": "phi3:mini-instruct",
}


@register_strategy("ollama")
class OllamaStrategy(BaseStrategy):
    """
    Ollama: easy local model serving with Metal/CUDA/CPU support.
    Best for: local dev, Apple Silicon, users who want simplicity.
    Drawback: no batched inference (1 request at a time).
    """

    def is_compatible(self, model, hardware) -> bool:
        """Return True for LLM and GGUF model types on any hardware.

        Ollama handles CPU, MPS, and CUDA through its own backend selection;
        it never requires a specific GPU to be present.

        Args:
            model: ModelProfile — compatible when model_type is "llm" or "gguf".
            hardware: HardwareProfile — accepted for any hardware configuration.

        Returns:
            True when the model type is llm or gguf; False otherwise.
        """
        if model.model_type not in ("llm", "gguf"):
            return False
        return True

    def estimate_feasibility(self, model, hardware) -> FeasibilityScore:
        from mlfit.core.memory import estimate_weights_gb

        weights_gb = estimate_weights_gb(model.num_parameters,
                                          model.effective_dtype)

        if hardware.gpu_backend == "mps":
            if weights_gb <= hardware.total_vram_gb * 0.85:
                return FeasibilityScore(0.95, "Excellent — native Metal acceleration", [])
            return FeasibilityScore(0.60, "Fits with unified memory", ["Large model"])

        if hardware.gpu_backend == "cuda":
            if weights_gb <= hardware.total_vram_gb * 0.85:
                return FeasibilityScore(0.65, "Good for local use, no batching",
                                        ["Sequential requests only"])
            return FeasibilityScore(0.40, "Model may not fully fit on GPU",
                                    ["Will use CPU offload"])

        ram_available = hardware.ram_gb * 0.6
        if weights_gb <= ram_available:
            return FeasibilityScore(0.55, "Works on CPU, will be slow",
                                    ["CPU inference only"])
        return FeasibilityScore(0.15, "Model too large for CPU RAM", [])

    def generate_config(self, model, hardware) -> BackendConfig:
        from mlfit.core.memory import estimate_weights_gb

        ctx = min(model.max_context_length, 4096)

        params = {
            "num_ctx": ctx,
            "num_thread": hardware.cpu_cores,
        }

        if hardware.gpu_backend in ("cuda", "rocm", "mps"):
            params["num_gpu"] = 999

        ollama_name = self._get_ollama_name(model.model_id)
        command = f"ollama run {ollama_name}"

        if ctx != 4096:
            command += f" --num-ctx {ctx}"

        weights_gb = estimate_weights_gb(model.num_parameters,
                                          model.effective_dtype)

        return BackendConfig(
            backend="ollama",
            model_id=model.model_id,
            params=params,
            command=command,
            estimated_vram_gb=weights_gb * 1.1,
            estimated_tps=self._estimate_tps(model, hardware),
            server_url="http://localhost:11434",
            server_command="ollama serve",
            health_path="/api/tags",
        )

    def format_key_settings(self, params: dict) -> str:
        """Format Ollama-specific params: context window and GPU layer count."""
        parts = []
        if "num_ctx" in params:
            parts.append(f"num_ctx={params['num_ctx']}")
        if "num_gpu" in params:
            parts.append(f"num_gpu={params['num_gpu']}")
        return ", ".join(parts) or "defaults"

    def _get_ollama_name(self, model_id: str) -> str:
        """Convert HF model ID to Ollama library name."""
        if model_id in OLLAMA_MODEL_MAP:
            return OLLAMA_MODEL_MAP[model_id]
        return model_id.split("/")[-1].lower().replace("-", ":").replace("instruct", "instruct")

    def _estimate_tps(self, model, hardware) -> float:
        """Ollama TPS estimates (no batching, single request)."""
        gpu_name = self._gpu_name(hardware)

        if hardware.gpu_backend == "mps":
            base = 60
        elif hardware.gpu_backend == "cuda":
            if "4090" in gpu_name:
                base = 50
            elif "3090" in gpu_name:
                base = 35
            else:
                base = 25
        elif hardware.has_avx512:
            base = 12
        elif hardware.has_avx2:
            base = 6
        else:
            base = 2

        return round(base * self._param_scale(model), 1)
