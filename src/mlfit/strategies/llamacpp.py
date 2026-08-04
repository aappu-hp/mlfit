from mlfit.strategies.base import BaseStrategy, BackendConfig, FeasibilityScore
from mlfit.strategies import register_strategy


@register_strategy("llamacpp")
class LlamaCppStrategy(BaseStrategy):
    """
    llama.cpp: efficient C++ inference with GGUF format.
    Best for: CPU inference, GGUF models, Metal on Apple Silicon,
              edge deployments, maximum control over parameters.
    """

    def is_compatible(self, model, hw) -> bool:
        return model.model_type in ("llm", "gguf")

    def estimate_feasibility(self, model, hw) -> FeasibilityScore:
        from mlfit.core.memory import estimate_weights_gb

        weights_gb = estimate_weights_gb(model.num_parameters,
                                          model.quant_type or model.dtype)

        if hw.gpu_backend == "mps":
            if weights_gb <= hw.total_vram_gb * 0.80:
                return FeasibilityScore(0.82, "Good — Metal acceleration available",
                                        ["Ollama is simpler for same performance"])
            return FeasibilityScore(0.50, "Fits with unified memory, may need quant", [])

        if hw.gpu_backend in ("cuda", "rocm"):
            if weights_gb <= hw.total_vram_gb * 0.85:
                return FeasibilityScore(0.52, "Works but vLLM is faster on CUDA",
                                        ["Consider vLLM for higher throughput"])
            return FeasibilityScore(0.35, "Will CPU offload layers that don't fit", [])

        ram_available = hw.ram_gb * 0.65
        if weights_gb <= ram_available:
            score = 0.80 if hw.has_avx512 else 0.70 if hw.has_avx2 else 0.55
            avx_note = "AVX512" if hw.has_avx512 else "AVX2" if hw.has_avx2 else "no AVX"
            return FeasibilityScore(score, f"Best CPU inference engine ({avx_note})", [])
        return FeasibilityScore(0.10, "Model too large for available RAM", [])

    def generate_config(self, model, hw) -> BackendConfig:
        """
        Produce the optimal llama-server configuration for the hardware.

        GPU backends get all layers offloaded (-ngl 999) and a larger context
        window. CPU-only builds get physical core count for threads and mlock
        to prevent model pages from being swapped out. MPS (Apple Silicon)
        adds a separate --threads-batch value for prompt-processing parallelism.

        Args:
            model: ModelProfile with model_type == "llm" or "gguf".
            hw: HardwareProfile describing available compute.

        Returns:
            BackendConfig with a ready-to-run llama-server command.
        """
        params = {}

        if hw.gpu_backend != "none":
            params["n_gpu_layers"] = 999
            params["ctx_size"] = min(model.max_context_length, 8192)
        else:
            params["n_gpu_layers"] = 0
            params["ctx_size"] = min(model.max_context_length, 4096)
            params["mlock"] = True

        params["threads"] = hw.cpu_cores
        params["batch_size"] = 512

        if hw.gpu_backend == "mps":
            params["threads_batch"] = hw.cpu_cores

        model_filename = self._get_model_filename(model)
        command = self._build_command(model_filename, params)

        from mlfit.core.memory import estimate_weights_gb
        weights_gb = estimate_weights_gb(model.num_parameters,
                                          model.quant_type or model.dtype)

        return BackendConfig(
            backend="llamacpp",
            model_id=model.model_id,
            params=params,
            command=command,
            estimated_vram_gb=0.0 if hw.gpu_backend == "none" else weights_gb,
            estimated_tps=self._estimate_tps(model, hw),
            server_url="http://localhost:8080",
            health_path="/health",
        )

    def format_key_settings(self, params: dict) -> str:
        """
        Format llama.cpp-specific params: GPU layers, context size, thread count.

        Args:
            params: BackendConfig.params dict.

        Returns:
            Compact string for the recommendations table.
        """
        parts = []
        if "n_gpu_layers" in params:
            parts.append(f"ngl={params['n_gpu_layers']}")
        if "ctx_size" in params:
            parts.append(f"ctx={params['ctx_size']}")
        if "threads" in params:
            parts.append(f"threads={params['threads']}")
        if "threads_batch" in params:
            parts.append(f"tb={params['threads_batch']}")
        return ", ".join(parts) or "defaults"

    def _get_model_filename(self, model) -> str:
        """Get the model filename for the llama-server command."""
        if model.model_type == "gguf" and model.model_id.endswith(".gguf"):
            return model.model_id
        name = model.model_id.split("/")[-1]
        quant = model.quant_type or "Q4_K_M"
        return f"{name}-{quant}.gguf"

    def _build_command(self, model_file: str, params: dict) -> str:
        """
        Build the llama-server CLI command from the resolved params dict.

        Args:
            model_file: Path or filename of the GGUF model.
            params: Resolved parameter dict from generate_config.

        Returns:
            Multi-line shell command string.
        """
        args = [f"llama-server -m {model_file}"]
        flag_map = {
            "n_gpu_layers": "ngl",
            "ctx_size": "c",
            "threads": "t",
            "threads_batch": "tb",
            "batch_size": "b",
            "mlock": "mlock",
        }
        for k, v in params.items():
            flag = flag_map.get(k, k.replace("_", "-"))
            if isinstance(v, bool):
                if v:
                    args.append(f"--{flag}")
            else:
                args.append(f"--{flag} {v}")
        args.append("--host 0.0.0.0 --port 8080")
        return " \\\n    ".join(args)

    def _estimate_tps(self, model, hw) -> float:
        params_b = model.num_parameters / 1e9
        gpu_name = hw.gpus[0].name.lower() if hw.gpus else ""

        if hw.gpu_backend == "mps":
            base = 55
        elif hw.gpu_backend in ("cuda", "rocm"):
            if "4090" in gpu_name:
                base = 40
            elif "3090" in gpu_name:
                base = 28
            else:
                base = 18
        elif hw.has_avx512:
            base = 12
        elif hw.has_avx2:
            base = 6
        else:
            base = 2

        scale = min(7.0 / max(params_b, 0.5), 1.0)
        return round(base * scale, 1)
