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
    ) -> RecommendResult:
        """
        Recommend optimal serving configurations.

        Args:
            backend: "auto" | "vllm" | "ollama" | "llamacpp" | "tgi"

        Returns:
            RecommendResult with ranked configs and quantized alternatives when model doesn't fit.
        """
        from mlfit.strategies import STRATEGY_REGISTRY, _load_all_strategies
        from mlfit.core.memory import check_fits

        _load_all_strategies()

        hardware = self.hardware
        model = self.model

        compatible = []
        for name, StrategyClass in STRATEGY_REGISTRY.items():
            if backend != "auto" and name != backend:
                continue
            strategy = StrategyClass()
            if strategy.is_compatible(model, hardware):
                config = strategy.generate_config(model, hardware)
                score = strategy.estimate_feasibility(model, hardware)
                compatible.append((score, config, strategy))
                logger.debug("Strategy %s: score=%.2f", name, score.score)

        compatible.sort(reverse=True, key=lambda x: x[0].score)
        logger.info(
            "Scored %d compatible strategies for %s", len(compatible), model.model_id
        )

        fits, estimated_vram, headroom = check_fits(model, hardware)

        quant_alternatives = []
        if not fits:
            from mlfit.quantization.advisor import find_quantized_alternatives
            quant_alternatives = find_quantized_alternatives(
                self.model_id, hardware
            )
            logger.info(
                "Model does not fit (needs %.1f GB); found %d quantized alternatives",
                estimated_vram,
                len(quant_alternatives),
            )

        return RecommendResult(
            hardware=hardware,
            model=model,
            configs=compatible,
            is_feasible=fits,
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

        hardware = self.hardware
        model = self.model
        concurrency_levels = concurrency or [1, 4, 16]

        strategy = self._select_strategy(backend, model, hardware)
        logger.info(
            "Profile starting: model=%s backend=%s concurrency=%s",
            model.model_id, strategy.__class__.__name__, concurrency_levels,
        )

        session = ProfileSession(
            model=model,
            hardware=hardware,
            strategy=strategy,
            concurrency_levels=concurrency_levels,
            progress_callback=progress_callback,
        )
        return session.run()

    async def profile_async(
        self,
        backend: str = "auto",
        concurrency: list = None,
        progress_callback=None,
    ):
        """
        Non-blocking version of profile() for use in async contexts.

        Runs the blocking profile() call in a thread pool via asyncio.to_thread,
        leaving the event loop free for other work during the (potentially long)
        profiling session.

        Args:
            backend: "auto" | "vllm" | "ollama" | "llamacpp" | "tgi"
            concurrency: List of concurrency levels, e.g. [1, 4, 16].
            progress_callback: Optional callable(phase, message, is_success).

        Returns:
            ProfilingResult — same as profile().
        """
        import asyncio
        return await asyncio.to_thread(self.profile, backend, concurrency, progress_callback)

    async def stream_profile(self, backend: str = "auto", concurrency: list = None):
        """
        Async generator that yields live progress updates during profiling.

        Bridges the synchronous progress_callback to an async for loop via an
        asyncio.Queue. Yields dicts with keys phase/message/is_success for each
        progress event, then a final dict with phase="complete" and result=ProfilingResult.

        Args:
            backend: "auto" | "vllm" | "ollama" | "llamacpp" | "tgi"
            concurrency: List of concurrency levels, e.g. [1, 4, 16].

        Yields:
            {"phase": str, "message": str, "is_success": bool} for each step.
            {"phase": "complete", "result": ProfilingResult} as the final item.
        """
        import asyncio

        loop = asyncio.get_event_loop()
        queue: asyncio.Queue = asyncio.Queue()
        sentinel = object()

        def on_progress(phase: str, message: str, is_success: bool) -> None:
            # call_soon_threadsafe is required because on_progress is called from
            # the worker thread created by asyncio.to_thread, and asyncio.Queue
            # is not thread-safe — Future.set_result() must run on the event loop thread.
            loop.call_soon_threadsafe(
                queue.put_nowait,
                {"phase": phase, "message": message, "is_success": is_success},
            )

        async def _run():
            result = await asyncio.to_thread(self.profile, backend, concurrency, on_progress)
            queue.put_nowait(sentinel)
            return result

        task = asyncio.create_task(_run())
        while True:
            item = await queue.get()
            if item is sentinel:
                break
            yield item

        result = await task
        yield {"phase": "complete", "result": result}

    def _select_strategy(self, backend: str, model, hardware):
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
            (cls().estimate_feasibility(model, hardware).score, cls())
            for cls in STRATEGY_REGISTRY.values()
            if cls().is_compatible(model, hardware)
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
