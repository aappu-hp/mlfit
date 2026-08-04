import logging

from mlfit.core.models import RecommendResult

logger = logging.getLogger(__name__)


class Advisor:
    """
    Universal model deployment advisor.

    Usage:
        advisor = Advisor("meta-llama/Llama-3-8B")
        result = advisor.recommend()
        print(result.best.command)
    """

    def __init__(self, model_id: str, gpuid: str = "all"):
        self.model_id = model_id
        self.gpuid = gpuid
        self._hardware = None
        self._model = None

    @property
    def hardware(self):
        """Lazily detect and cache the HardwareProfile for this session."""
        if self._hardware is None:
            from mlfit.detectors import detect_hardware
            logger.debug("Detecting hardware (gpuid=%s)", self.gpuid)
            self._hardware = detect_hardware(self.gpuid)
            logger.info(
                "Hardware detected: backend=%s, total_vram_gb=%.1f",
                self._hardware.gpu_backend,
                self._hardware.total_vram_gb,
            )
        return self._hardware

    @property
    def model(self):
        """Lazily fetch and cache the ModelProfile for this session."""
        if self._model is None:
            from mlfit.analyzers import analyze_model
            logger.debug("Analyzing model: %s", self.model_id)
            self._model = analyze_model(self.model_id)
            logger.info(
                "Model analyzed: id=%s, params=%.2fB, dtype=%s, size_gb=%.2f",
                self._model.model_id,
                self._model.num_parameters / 1e9,
                self._model.dtype,
                self._model.estimated_size_gb,
            )
        return self._model

    def recommend(
        self,
        backend: str = "auto",
        include_quant: bool = True,
    ) -> RecommendResult:
        """
        Recommend optimal serving configurations.

        Args:
            backend: "auto" | "vllm" | "ollama" | "llamacpp" | "tgi"
            include_quant: If True and model doesn't fit, search for quantized alternatives

        Returns:
            RecommendResult with ranked configs and alternatives
        """
        from mlfit.strategies import STRATEGY_REGISTRY, _load_all_strategies
        from mlfit.core.memory import check_fits

        _load_all_strategies()

        hw = self.hardware
        model = self.model

        compatible = []
        for name, StrategyClass in STRATEGY_REGISTRY.items():
            if backend != "auto" and name != backend:
                continue
            strategy = StrategyClass()
            if strategy.is_compatible(model, hw):
                config = strategy.generate_config(model, hw)
                score = strategy.estimate_feasibility(model, hw)
                compatible.append((score, config, strategy))
                logger.debug("Strategy %s: score=%.2f", name, score.score)

        compatible.sort(reverse=True, key=lambda x: x[0].score)
        logger.info(
            "Scored %d compatible strategies for %s", len(compatible), model.model_id
        )

        fits, estimated_vram, headroom = check_fits(model, hw)

        quant_alternatives = []
        if not fits and include_quant:
            from mlfit.quantization.advisor import find_quantized_alternatives
            quant_alternatives = find_quantized_alternatives(
                self.model_id, hw
            )
            logger.info(
                "Model does not fit (needs %.1f GB); found %d quantized alternatives",
                estimated_vram,
                len(quant_alternatives),
            )

        return RecommendResult(
            hardware=hw,
            model=model,
            configs=compatible,
            fits=fits,
            estimated_vram_gb=estimated_vram,
            headroom_gb=headroom,
            quantized_alternatives=quant_alternatives,
        )

    def profile(
        self,
        backend: str = "auto",
        concurrency: list = None,
        progress_callback=None,
    ):
        """
        Profile actual memory usage and throughput by loading the model and benchmarking.

        Downloads model weights if not cached. Starts the backend server as a subprocess,
        measures peak VRAM during load, TTFT latency, and TPS at multiple concurrency levels.

        Args:
            backend: "auto" selects the highest-scoring compatible strategy.
                     Pass "vllm", "ollama", "llamacpp", or "tgi" to force a backend.
            concurrency: List of concurrency levels to benchmark, e.g. [1, 4, 16].
                         Defaults to [1, 4, 16] when None.
            progress_callback: Optional callable(phase, message, is_success) invoked
                               at each step so the CLI can display live progress.

        Returns:
            ProfilingResult with real VRAM, TTFT, and TPS measurements.

        Raises:
            RuntimeError: If no compatible backend is found or no config loads cleanly.
        """
        from mlfit.strategies import STRATEGY_REGISTRY, _load_all_strategies
        from mlfit.profiler.session import ProfileSession

        _load_all_strategies()

        hw = self.hardware
        model = self.model
        concurrency_levels = concurrency or [1, 4, 16]

        strategy = self._select_strategy(backend, model, hw)
        logger.info(
            "Profile starting: model=%s backend=%s concurrency=%s",
            model.model_id, strategy.__class__.__name__, concurrency_levels,
        )

        session = ProfileSession(
            model=model,
            hardware=hw,
            strategy=strategy,
            concurrency_levels=concurrency_levels,
            progress_callback=progress_callback,
        )
        return session.run()

    def _select_strategy(self, backend: str, model, hw):
        from mlfit.strategies import STRATEGY_REGISTRY

        if backend != "auto":
            cls = STRATEGY_REGISTRY.get(backend)
            if cls is None:
                raise ValueError(
                    f"Unknown backend '{backend}'. "
                    f"Available: {', '.join(STRATEGY_REGISTRY)}"
                )
            return cls()

        compatible = [
            (cls().estimate_feasibility(model, hw).score, cls())
            for cls in STRATEGY_REGISTRY.values()
            if cls().is_compatible(model, hw)
        ]
        if not compatible:
            raise RuntimeError(
                f"No compatible backend found for '{model.model_id}' on this hardware. "
                "Run `mlfit recommend` to see why."
            )
        compatible.sort(reverse=True, key=lambda x: x[0])
        selected = compatible[0][1]
        logger.debug(
            "Auto-selected strategy: %s (score=%.2f)",
            selected.__class__.__name__, compatible[0][0],
        )
        return selected
