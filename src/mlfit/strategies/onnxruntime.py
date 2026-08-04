from mlfit.strategies.base import BaseStrategy, BackendConfig, FeasibilityScore
from mlfit.strategies import register_strategy


@register_strategy("onnxruntime")
class ONNXRuntimeStrategy(BaseStrategy):
    """
    ONNX Runtime: cross-platform, high-performance inference engine.

    Best for: ONNX model files, CPU-optimised inference with AVX2/AVX512,
              cross-platform deployment, and GPU acceleration via CUDA or CoreML.
    Requires: onnxruntime (CPU) or onnxruntime-gpu (CUDA) package.
    """

    def is_compatible(self, model, hw) -> bool:
        """Return True only for ONNX model files."""
        return model.model_type == "onnx"

    def estimate_feasibility(self, model, hw) -> FeasibilityScore:
        """
        Score ONNX Runtime suitability for the given hardware.

        CUDA GPU scores highest because the CUDA execution provider gives
        the best inference speed. CoreML on MPS is the next best. CPU
        scores vary with AVX capability because ONNX Runtime auto-enables
        AVX2/AVX512 kernels when available.

        Args:
            model: ModelProfile with model_type == "onnx".
            hw: HardwareProfile describing available compute.

        Returns:
            FeasibilityScore in range 0.45–0.88.
        """
        if hw.gpu_backend == "cuda":
            return FeasibilityScore(
                0.88,
                "Excellent — CUDA execution provider available",
                [],
            )
        if hw.gpu_backend == "mps":
            return FeasibilityScore(
                0.80,
                "Good — CoreML execution provider via MPS",
                [],
            )
        if hw.has_avx512:
            return FeasibilityScore(
                0.72,
                "Good — CPU inference with AVX512 kernels",
                [],
            )
        if hw.has_avx2:
            return FeasibilityScore(
                0.65,
                "Moderate — CPU inference with AVX2 kernels",
                [],
            )
        return FeasibilityScore(
            0.45,
            "Baseline CPU inference — no AVX extensions detected",
            ["Performance will be limited without AVX2/AVX512"],
        )

    def generate_config(self, model, hw) -> BackendConfig:
        """
        Produce an ONNX Runtime InferenceSession configuration.

        Selects the best available execution provider for the hardware
        and sets intra/inter-op thread counts from the CPU core count.
        The generated command is a Python code snippet because ONNX Runtime
        is a library API rather than a CLI server.

        Args:
            model: ModelProfile with model_type == "onnx".
            hw: HardwareProfile describing available compute.

        Returns:
            BackendConfig with a Python snippet as the command field.
        """
        providers = self._select_providers(hw)
        intra_threads = hw.cpu_cores
        inter_threads = min(hw.cpu_cores // 4, 4) if hw.cpu_cores >= 8 else 1

        params = {
            "providers": providers,
            "intra_op_num_threads": intra_threads,
            "inter_op_num_threads": inter_threads,
        }

        command = self._build_code_snippet(model.model_id, providers, intra_threads, inter_threads)

        vram_gb = model.estimated_size_gb if hw.gpu_backend in ("cuda", "mps") else 0.0

        return BackendConfig(
            backend="onnxruntime",
            model_id=model.model_id,
            params=params,
            command=command,
            estimated_vram_gb=vram_gb,
            estimated_tps=self._estimate_throughput(model, hw),
        )

    def format_key_settings(self, params: dict) -> str:
        """
        Format ONNX Runtime key parameters: provider and thread counts.

        Args:
            params: BackendConfig.params dict.

        Returns:
            Compact string for the recommendations table.
        """
        parts = []
        if "providers" in params:
            provider = params["providers"][0].replace("ExecutionProvider", "")
            parts.append(f"provider={provider}")
        if "intra_op_num_threads" in params:
            parts.append(f"threads={params['intra_op_num_threads']}")
        return ", ".join(parts) or "defaults"

    def _select_providers(self, hw) -> list:
        """
        Return the ordered execution provider list for the hardware.

        ONNX Runtime tries providers in order and falls back to the next
        available one, so the list order encodes priority.

        Args:
            hw: HardwareProfile.

        Returns:
            List of provider name strings.
        """
        if hw.gpu_backend == "cuda":
            return ["CUDAExecutionProvider", "CPUExecutionProvider"]
        if hw.gpu_backend == "mps":
            return ["CoreMLExecutionProvider", "CPUExecutionProvider"]
        return ["CPUExecutionProvider"]

    def _build_code_snippet(
        self, model_path: str, providers: list, intra_threads: int, inter_threads: int
    ) -> str:
        """
        Build the Python code snippet that users can run directly.

        Args:
            model_path: Path to the .onnx file.
            providers: Ordered list of execution provider names.
            intra_threads: Parallelism within a single op.
            inter_threads: Parallelism across independent ops.

        Returns:
            Multi-line Python string.
        """
        providers_repr = repr(providers)
        return (
            f"import onnxruntime as ort\n\n"
            f"options = ort.SessionOptions()\n"
            f"options.intra_op_num_threads = {intra_threads}\n"
            f"options.inter_op_num_threads = {inter_threads}\n"
            f"session = ort.InferenceSession(\n"
            f'    "{model_path}",\n'
            f"    sess_options=options,\n"
            f"    providers={providers_repr},\n"
            f")\n"
            f"outputs = session.run(None, {{\"input\": input_data}})"
        )

    def _estimate_throughput(self, model, hw) -> float:
        """
        Estimate inferences per second based on hardware capability.

        Args:
            model: ModelProfile with estimated_size_gb populated.
            hw: HardwareProfile.

        Returns:
            Estimated throughput as a float (inferences/second).
        """
        if hw.gpu_backend == "cuda":
            return 25000.0
        if hw.gpu_backend == "mps":
            return 15000.0
        if hw.has_avx512:
            return 18000.0
        if hw.has_avx2:
            return 10000.0
        return 4000.0
