from mlfit.strategies.base import BaseStrategy, BackendConfig, FeasibilityScore
from mlfit.strategies import register_strategy

_SERVER_URL = "http://localhost:8000"
_MEM_UTIL_CANDIDATES = [0.70, 0.80, 0.85, 0.90]


@register_strategy("vllm")
class VLLMStrategy(BaseStrategy):
    """
    vLLM: PagedAttention-based LLM serving.
    Best for: NVIDIA CUDA, multi-GPU, high concurrency.
    Not supported on: Apple Silicon MPS, CPU-only.
    """

    def is_compatible(self, model, hardware) -> bool:
        """
        Check whether vLLM can run this model on this hardware.

        Args:
            model: ModelProfile describing architecture and size.
            hardware: HardwareProfile describing available resources.

        Returns:
            True only on CUDA/ROCm with at least 4 GB VRAM and an LLM model type.
        """
        if hardware.gpu_backend not in ("cuda", "rocm"):
            return False
        if model.model_type not in ("llm", "gguf"):
            return False
        if hardware.total_vram_gb < 4.0:
            return False
        return True

    def estimate_feasibility(self, model, hardware) -> FeasibilityScore:
        """
        Score how well vLLM suits this model/hardware combination.

        Args:
            model: ModelProfile describing architecture and size.
            hardware: HardwareProfile describing available resources.

        Returns:
            FeasibilityScore between 0.0 and 1.0 with a human-readable reason.
        """
        from mlfit.core.memory import estimate_vram_gb
        needed = estimate_vram_gb(model, max_seq_len=4096, max_batch=1)
        ratio = needed / max(hardware.total_vram_gb, 1.0)

        if ratio > 1.2:
            return FeasibilityScore(0.0, "Model too large even with aggressive settings", [])
        if ratio > 0.95:
            return FeasibilityScore(0.3, "Very tight fit, enforce_eager needed",
                                    ["May OOM at high concurrency"])
        if ratio > 0.80:
            return FeasibilityScore(0.75, "Good fit with conservative settings", [])

        score = 0.94 if hardware.gpu_backend == "cuda" else 0.80
        return FeasibilityScore(score, "Excellent fit, can use aggressive settings", [])

    def generate_config(self, model, hardware) -> BackendConfig:
        """
        Produce the optimal BackendConfig by auto-selecting memory utilisation.

        Args:
            model: ModelProfile describing architecture and size.
            hardware: HardwareProfile describing available resources.

        Returns:
            BackendConfig with resolved vLLM parameters and a ready-to-run command.
        """
        from mlfit.core.memory import estimate_weights_gb
        weights_gb = estimate_weights_gb(model.num_parameters, model.effective_dtype)
        vram_ratio = weights_gb / max(hardware.total_vram_gb, 1.0)

        if vram_ratio > 0.90:
            mem_util = 0.72
        elif vram_ratio > 0.78:
            mem_util = 0.82
        else:
            mem_util = 0.90

        return self._build_config(model, hardware, mem_util)

    def generate_config_candidates(self, model, hardware) -> list:
        """
        Return configs from conservative to aggressive for iterative profiling search.

        The profiler starts with the first candidate and escalates until one
        successfully loads without OOM.

        Args:
            model: ModelProfile describing architecture and size.
            hardware: HardwareProfile describing available resources.

        Returns:
            List of BackendConfigs ordered by increasing memory utilisation.
        """
        return [self._build_config(model, hardware, mem_util)
                for mem_util in _MEM_UTIL_CANDIDATES]

    def format_key_settings(self, params: dict) -> str:
        """
        Format vLLM-specific params: memory utilisation, context length, parallelism.

        Args:
            params: The BackendConfig.params dict for this configuration.

        Returns:
            Compact string for display in the recommendations table.
        """
        parts = []
        if "gpu_memory_utilization" in params:
            parts.append(f"mem={params['gpu_memory_utilization']}")
        if "max_model_len" in params:
            parts.append(f"len={params['max_model_len']}")
        if "max_num_seqs" in params:
            parts.append(f"seqs={params['max_num_seqs']}")
        if params.get("tensor_parallel_size", 1) > 1:
            parts.append(f"tp={params['tensor_parallel_size']}")
        if params.get("enforce_eager"):
            parts.append("eager")
        return ", ".join(parts) or "defaults"

    def _build_config(self, model, hardware, mem_util: float) -> BackendConfig:
        from mlfit.detectors import get_tensor_parallel_size

        enforce_eager = mem_util < 0.75

        max_model_len = min(model.max_context_length, 8192)
        tp_size = get_tensor_parallel_size(hardware.gpus, model.num_attention_heads)
        max_num_seqs = self._estimate_max_seqs(model, hardware, mem_util, max_model_len, tp_size)

        params = {
            "gpu_memory_utilization": mem_util,
            "max_model_len": max_model_len,
            "tensor_parallel_size": tp_size,
            "max_num_seqs": max_num_seqs,
        }
        if enforce_eager:
            params["enforce_eager"] = True

        command = self._build_command(model.model_id, params)

        return BackendConfig(
            backend="vllm",
            model_id=model.model_id,
            params=params,
            command=command,
            estimated_vram_gb=hardware.total_vram_gb * mem_util,
            estimated_tps=self._estimate_tps(model, hardware),
            server_url=_SERVER_URL,
            health_path="/health",
        )

    def _build_command(self, model_id: str, params: dict) -> str:
        """Build the vllm serve CLI command from resolved params.

        Args:
            model_id: HuggingFace model identifier.
            params: Resolved parameter dict from _build_config.

        Returns:
            Multi-line shell command string ready to copy-paste.
        """
        args = [f"vllm serve {model_id}"]
        for k, v in params.items():
            flag = "--" + k.replace("_", "-")
            if isinstance(v, bool):
                if v:
                    args.append(flag)
            else:
                args.append(f"{flag} {v}")
        return " \\\n    ".join(args)

    def _estimate_max_seqs(self, model, hardware, mem_util: float, max_len: int, tp_size: int) -> int:
        """Estimate max concurrent sequences that fit within available VRAM.

        Args:
            model: ModelProfile used for weight and KV cache size calculation.
            hardware: HardwareProfile with total_vram_gb.
            mem_util: Fraction of VRAM vLLM is allowed to use (e.g. 0.85).
            max_len: Maximum sequence length in tokens.
            tp_size: Tensor parallel degree (number of GPUs).

        Returns:
            Clamped integer in [4, 256] representing max_num_seqs.
        """
        from mlfit.core.memory import estimate_weights_gb, BYTES_PER_PARAM
        weights_gb = estimate_weights_gb(model.num_parameters, model.effective_dtype)
        available = hardware.total_vram_gb * mem_util - weights_gb - 0.75

        bpp = BYTES_PER_PARAM.get(model.effective_dtype, 2.0)
        heads = max(model.num_key_value_heads, 1)
        head_dim = max(model.hidden_size // max(model.num_attention_heads, 1), 1)
        per_seq_gb = (2 * model.num_hidden_layers * heads * head_dim * max_len * bpp) / 1e9

        if per_seq_gb <= 0:
            return 32
        raw = int(available / per_seq_gb)
        return max(4, min(256, raw))

    def _estimate_tps(self, model, hardware) -> float:
        gpu_name = self._gpu_name(hardware)

        if "h100" in gpu_name:
            base_tps = 200
        elif "a100" in gpu_name:
            base_tps = 110
        elif "4090" in gpu_name or "4080" in gpu_name:
            base_tps = 85
        elif "3090" in gpu_name or "3080" in gpu_name:
            base_tps = 50
        elif "a10" in gpu_name or "a16" in gpu_name:
            base_tps = 60
        else:
            base_tps = 30

        return round(base_tps * self._param_scale(model), 1)
