import datetime
import logging
import time
from typing import Callable, Optional

from mlfit.core.models import BenchmarkPoint, ProfilingResult
from mlfit.strategies.base import BackendConfig, BaseStrategy

logger = logging.getLogger(__name__)

ProgressCallback = Callable[[str, str, bool], None]


class ConfigSearcher:
    """
    Iterates through a strategy's config candidates from conservative to aggressive,
    returning the most memory-efficient config that successfully loads.

    Uses Template Method: the strategy decides what candidates to generate;
    this class decides how to evaluate them.
    """

    def __init__(
        self,
        strategy: BaseStrategy,
        progress_callback: Optional[ProgressCallback] = None,
    ):
        """
        Args:
            strategy: The backend strategy supplying candidate configs.
            progress_callback: Optional callable(phase, message, is_success).
        """
        self._strategy = strategy
        self._callback = progress_callback or _noop_callback

    def find_optimal(self, model, hardware) -> Optional[BackendConfig]:
        """
        Return the most aggressive working config, or None if nothing loads.

        Iterates candidates in order (least → most aggressive). The last
        successfully loaded config wins — this maximises VRAM utilisation
        without triggering OOM.

        Args:
            model: ModelProfile for the model being profiled.
            hardware: HardwareProfile describing available resources.

        Returns:
            The most aggressive BackendConfig that loaded cleanly, or None.
        """
        from mlfit.profiler.server import ServerManager

        candidates = self._strategy.generate_config_candidates(model, hardware)
        logger.info(
            "Config search: trying %d candidates for %s",
            len(candidates), model.model_id,
        )

        best: Optional[BackendConfig] = None

        for i, config in enumerate(candidates):
            mem_util = config.params.get("gpu_memory_utilization", "auto")
            max_len = config.params.get("max_model_len", "auto")
            self._callback(
                "searching",
                f"Trying mem={mem_util}, len={max_len}  ({i + 1}/{len(candidates)})",
                True,
            )

            with ServerManager(config, timeout_s=90) as server:
                if server.wait_until_ready():
                    best = config
                    self._callback("searching", f"✓ Loaded  ({mem_util})", True)
                    logger.info(
                        "Config candidate %d/%d succeeded: mem=%s",
                        i + 1, len(candidates), mem_util,
                    )
                else:
                    self._callback(
                        "searching",
                        f"✗ OOM or timeout at mem={mem_util} — reducing",
                        False,
                    )
                    logger.warning(
                        "Config candidate %d/%d failed: mem=%s",
                        i + 1, len(candidates), mem_util,
                    )

        return best


class ProfileSession:
    """
    Orchestrates the four-phase profiling pipeline.

    Phase 1 — Config search: finds the most aggressive config that loads cleanly.
    Phase 2 — Warmup:        runs a fixed number of dummy requests to prime GPU kernels.
    Phase 3 — Latency:       measures TTFT p50/p95 via streaming requests.
    Phase 4 — Throughput:    measures TPS at each requested concurrency level.

    Uses Observer pattern: emits progress events via ProgressCallback so the
    CLI can display live status without coupling to Rich directly.
    """

    _WARMUP_REQUESTS = 5
    _TTFT_SAMPLES = 20

    def __init__(
        self,
        model,
        hardware,
        strategy: BaseStrategy,
        concurrency_levels: list,
        progress_callback: Optional[ProgressCallback] = None,
    ):
        """
        Args:
            model: ModelProfile for the model being profiled.
            hardware: HardwareProfile describing available resources.
            strategy: Backend strategy to use for config generation.
            concurrency_levels: Concurrency values to benchmark (e.g. [1, 4, 16]).
            progress_callback: Optional callable(phase, message, is_success).
        """
        self._model = model
        self._hardware = hardware
        self._strategy = strategy
        self._concurrency_levels = concurrency_levels
        self._callback = progress_callback or _noop_callback

    def run(self) -> ProfilingResult:
        """
        Execute all four profiling phases and return the measured results.

        Returns:
            ProfilingResult with real VRAM, TTFT, and TPS measurements.

        Raises:
            RuntimeError: If no config successfully loads (model too large).
        """
        from mlfit.profiler.memory import MemoryTracker
        from mlfit.profiler.server import ServerManager
        from mlfit.profiler.latency import measure_ttft
        from mlfit.profiler.throughput import measure_tps

        start_time = time.monotonic()

        self._callback("searching", "Finding optimal config...", True)
        searcher = ConfigSearcher(self._strategy, self._callback)
        optimal_config = searcher.find_optimal(self._model, self._hardware)

        if optimal_config is None:
            raise RuntimeError(
                f"No configuration for '{self._model.model_id}' loaded successfully. "
                "The model may be too large for your hardware. "
                "Try `mlfit recommend` with --quant for smaller alternatives."
            )

        logger.info("Optimal config found, starting benchmark run")
        benchmark_points: list[BenchmarkPoint] = []

        with MemoryTracker() as tracker:
            with ServerManager(optimal_config) as server:
                if not server.wait_until_ready():
                    raise RuntimeError(
                        f"Server for '{self._model.model_id}' failed to start for benchmarking."
                    )

                self._callback(
                    "warmup",
                    f"Running {self._WARMUP_REQUESTS} warmup requests...",
                    True,
                )
                self._run_warmup(server.base_url)
                self._callback("warmup", "Warmup complete", True)

                self._callback("benchmarking", "Measuring latency (TTFT)...", True)
                latency = measure_ttft(
                    server.base_url,
                    self._model.model_id,
                    num_samples=self._TTFT_SAMPLES,
                )
                self._callback(
                    "benchmarking",
                    f"TTFT p50={latency.p50_ms:.0f} ms  p95={latency.p95_ms:.0f} ms",
                    True,
                )

                for level in self._concurrency_levels:
                    self._callback(
                        "benchmarking",
                        f"Measuring throughput at concurrency={level}...",
                        True,
                    )
                    throughput = measure_tps(
                        server.base_url, self._model.model_id, concurrency=level
                    )
                    benchmark_points.append(BenchmarkPoint(
                        concurrency=level,
                        tps=throughput.tps,
                        ttft_p50_ms=latency.p50_ms,
                        ttft_p95_ms=latency.p95_ms,
                    ))
                    self._callback(
                        "benchmarking",
                        f"Concurrency={level}: {throughput.tps:.1f} t/s",
                        True,
                    )

        elapsed = time.monotonic() - start_time
        timestamp = (
            datetime.datetime.now(datetime.timezone.utc)
            .isoformat()
            .replace("+00:00", "Z")
        )

        logger.info(
            "Profiling complete for %s in %.1fs", self._model.model_id, elapsed
        )

        return ProfilingResult(
            model_id=self._model.model_id,
            backend=optimal_config.backend,
            hardware=self._hardware,
            final_config=optimal_config,
            peak_vram_gb=tracker.peak_vram_gb(),
            benchmark_points=benchmark_points,
            time_elapsed_s=elapsed,
            timestamp=timestamp,
        )

    def _run_warmup(self, base_url: str) -> None:
        import httpx

        url = f"{base_url}/v1/completions"
        with httpx.Client(timeout=30.0) as client:
            for i in range(self._WARMUP_REQUESTS):
                try:
                    client.post(url, json={
                        "model": self._model.model_id,
                        "prompt": "Hello",
                        "max_tokens": 10,
                    })
                    logger.debug(
                        "Warmup request %d/%d complete", i + 1, self._WARMUP_REQUESTS
                    )
                except Exception as exc:
                    logger.warning("Warmup request %d failed: %s", i + 1, exc)


def _noop_callback(phase: str, message: str, is_success: bool) -> None:
    pass
