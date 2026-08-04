from mlfit.strategies.base import BaseStrategy, BackendConfig, FeasibilityScore
from mlfit.strategies import register_strategy


@register_strategy("tgi")
class TGIStrategy(BaseStrategy):
    """
    TGI: HuggingFace Text Generation Inference.
    Best for: production HF model serving, Inference Endpoints compatibility.
    Requires: Docker + NVIDIA CUDA GPU.
    """

    def is_compatible(self, model, hw) -> bool:
        if hw.gpu_backend != "cuda":
            return False
        return model.model_type == "llm"

    def estimate_feasibility(self, model, hw) -> FeasibilityScore:
        from mlfit.core.memory import estimate_vram_gb
        needed = estimate_vram_gb(model, max_seq_len=4096, max_batch=8)
        ratio = needed / max(hw.total_vram_gb, 1.0)

        if ratio > 1.1:
            return FeasibilityScore(0.0, "Model too large for available VRAM", [])
        if ratio > 0.90:
            return FeasibilityScore(0.50, "Tight fit, may need quantization",
                                    ["Consider --quantize bitsandbytes"])
        if ratio > 0.70:
            return FeasibilityScore(0.72, "Good fit with standard settings", [])
        return FeasibilityScore(0.78, "Excellent fit, HF-native serving", [])

    def generate_config(self, model, hw) -> BackendConfig:
        """
        Produce the optimal TGI Docker command for the model/hardware pair.

        Sets tensor parallel shards from GPU count, applies --quantize when
        the model's quant_type is known (AWQ/GPTQ) or when the model barely
        fits and bitsandbytes int8 quantization can reduce memory pressure.

        Args:
            model: ModelProfile with model_type == "llm".
            hw: HardwareProfile with gpu_backend == "cuda".

        Returns:
            BackendConfig with a Docker run command.
        """
        from mlfit.detectors import get_tensor_parallel_size
        from mlfit.core.memory import estimate_vram_gb

        tp_size = get_tensor_parallel_size(hw.gpus, model.num_attention_heads)

        max_input = min(model.max_context_length // 2, 4096)
        max_total = min(model.max_context_length, 8192)

        vram_est = estimate_vram_gb(model, max_seq_len=max_total, max_batch=8)
        ratio = vram_est / max(hw.total_vram_gb, 1.0)

        from mlfit.core.memory import estimate_weights_gb
        weights_gb = estimate_weights_gb(model.num_parameters, model.dtype)
        weights_ratio = weights_gb / max(hw.total_vram_gb, 1.0)

        quantize_flag = self._resolve_quantize_flag(model, weights_ratio)

        params = {
            "max_input_length": max_input,
            "max_total_tokens": max_total,
            "num_shard": tp_size,
        }
        if quantize_flag:
            params["quantize"] = quantize_flag

        gpus = ",".join(str(g.index) for g in hw.gpus)
        cmd_parts = [
            f"docker run --gpus '\"device={gpus}\"'",
            f"    -v $HF_HOME:/data",
            f"    -p 8080:80",
            f"    ghcr.io/huggingface/text-generation-inference:latest",
            f"    --model-id {model.model_id}",
            f"    --num-shard {tp_size}",
            f"    --max-input-length {max_input}",
            f"    --max-total-tokens {max_total}",
        ]
        if quantize_flag:
            cmd_parts.append(f"    --quantize {quantize_flag}")

        cmd = " \\\n".join(cmd_parts)

        return BackendConfig(
            backend="tgi",
            model_id=model.model_id,
            params=params,
            command=cmd,
            estimated_vram_gb=vram_est,
            estimated_tps=self._estimate_tps(model, hw),
            server_url="http://localhost:8080",
            health_path="/health",
        )

    def _resolve_quantize_flag(self, model, vram_ratio: float) -> str:
        """
        Determine the --quantize flag value based on model quant_type and fit.

        AWQ and GPTQ models already carry their quantization in the weights,
        so TGI needs to know the format. When the model barely fits (ratio
        above 0.85) and no format is specified, bitsandbytes int8 can reduce
        peak VRAM enough to avoid OOM.

        Args:
            model: ModelProfile — quant_type may be "AWQ", "GPTQ", or None.
            vram_ratio: Model weights GB divided by available VRAM GB.

        Returns:
            Quantize flag string, or empty string if none is needed.
        """
        if model.quant_type == "AWQ":
            return "awq"
        if model.quant_type == "GPTQ":
            return "gptq"
        if vram_ratio > 0.85:
            return "bitsandbytes"
        return ""

    def format_key_settings(self, params: dict) -> str:
        """Format TGI-specific params: shard count and token limits."""
        parts = []
        if "num_shard" in params:
            parts.append(f"shards={params['num_shard']}")
        if "max_total_tokens" in params:
            parts.append(f"max-tokens={params['max_total_tokens']}")
        return ", ".join(parts) or "defaults"

    def _estimate_tps(self, model, hw) -> float:
        """TGI is roughly ~15% slower than vLLM for single requests."""
        gpu_name = hw.gpus[0].name.lower() if hw.gpus else ""
        params_b = model.num_parameters / 1e9

        if "h100" in gpu_name:
            base = 170
        elif "a100" in gpu_name:
            base = 90
        elif "4090" in gpu_name:
            base = 72
        elif "3090" in gpu_name:
            base = 42
        else:
            base = 25

        scale = min(7.0 / max(params_b, 0.5), 1.0)
        return round(base * scale, 1)
