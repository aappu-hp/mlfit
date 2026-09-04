import unittest
from unittest.mock import MagicMock
from mlfit.core.models import ProfilingResult, BenchmarkPoint


def _make_result(benchmark_points):
    hw = MagicMock()
    hw.gpus = []
    hw.gpu_backend = "none"
    hw.ram_gb = 16.0
    cfg = MagicMock()
    cfg.params = {}
    cfg.command = ""
    return ProfilingResult(
        model_id="test/model",
        backend="vllm",
        hardware=hw,
        final_config=cfg,
        peak_vram_gb=0.0,
        benchmark_points=benchmark_points,
        time_elapsed_s=10.0,
        timestamp="2026-09-04T00:00:00Z",
    )


class TestTpsAtConcurrency(unittest.TestCase):
    def setUp(self):
        self.result = _make_result([
            BenchmarkPoint(concurrency=1, tps=45.2, ttft_p50_ms=120.0, ttft_p95_ms=180.0),
            BenchmarkPoint(concurrency=4, tps=198.7, ttft_p50_ms=95.0, ttft_p95_ms=140.0),
            BenchmarkPoint(concurrency=16, tps=312.5, ttft_p50_ms=200.0, ttft_p95_ms=310.0),
        ])

    def test_tps_at_concurrency_1(self):
        assert self.result.tps_at_concurrency(1) == 45.2

    def test_tps_at_concurrency_4(self):
        assert self.result.tps_at_concurrency(4) == 198.7

    def test_tps_at_concurrency_16(self):
        assert self.result.tps_at_concurrency(16) == 312.5

    def test_missing_concurrency_raises_value_error(self):
        with self.assertRaises(ValueError) as ctx:
            self.result.tps_at_concurrency(8)
        assert "8" in str(ctx.exception)

    def test_error_message_lists_available_levels(self):
        with self.assertRaises(ValueError) as ctx:
            self.result.tps_at_concurrency(32)
        msg = str(ctx.exception)
        assert "1" in msg
        assert "4" in msg
        assert "16" in msg

    def test_empty_benchmark_points_raises(self):
        empty = _make_result([])
        with self.assertRaises(ValueError):
            empty.tps_at_concurrency(1)
