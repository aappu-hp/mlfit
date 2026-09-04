"""Tests for Advisor async methods — no GPU or network required."""
import unittest
from unittest.mock import MagicMock, patch
from mlfit.core.advisor import Advisor
from mlfit.core.models import ProfilingResult, BenchmarkPoint


def _dummy_profiling_result():
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
        peak_vram_gb=12.5,
        benchmark_points=[
            BenchmarkPoint(concurrency=1, tps=45.2, ttft_p50_ms=120.0, ttft_p95_ms=180.0),
            BenchmarkPoint(concurrency=4, tps=198.7, ttft_p50_ms=95.0, ttft_p95_ms=140.0),
        ],
        time_elapsed_s=30.0,
        timestamp="2026-09-04T00:00:00Z",
    )


class TestProfileAsync(unittest.IsolatedAsyncioTestCase):
    async def test_profile_async_returns_profiling_result(self):
        expected = _dummy_profiling_result()
        advisor = Advisor("test/model")
        with patch.object(advisor, "profile", return_value=expected):
            result = await advisor.profile_async()
        assert result is expected

    async def test_profile_async_forwards_backend_arg(self):
        expected = _dummy_profiling_result()
        advisor = Advisor("test/model")
        with patch.object(advisor, "profile", return_value=expected) as mock_profile:
            await advisor.profile_async(backend="vllm")
        mock_profile.assert_called_once_with("vllm", None, None)

    async def test_profile_async_forwards_concurrency_arg(self):
        expected = _dummy_profiling_result()
        advisor = Advisor("test/model")
        with patch.object(advisor, "profile", return_value=expected) as mock_profile:
            await advisor.profile_async(backend="vllm", concurrency=[1, 4])
        mock_profile.assert_called_once_with("vllm", [1, 4], None)


class TestStreamProfile(unittest.IsolatedAsyncioTestCase):
    async def test_stream_profile_yields_complete_as_last_item(self):
        expected = _dummy_profiling_result()
        advisor = Advisor("test/model")

        def fake_profile(backend, concurrency, progress_callback):
            progress_callback("loading", "Downloading model", True)
            progress_callback("benchmarking", "Running benchmark", True)
            return expected

        with patch.object(advisor, "profile", side_effect=fake_profile):
            items = []
            async for update in advisor.stream_profile():
                items.append(update)

        assert items[-1]["phase"] == "complete"
        assert items[-1]["result"] is expected

    async def test_stream_profile_yields_progress_updates_before_complete(self):
        expected = _dummy_profiling_result()
        advisor = Advisor("test/model")

        def fake_profile(backend, concurrency, progress_callback):
            progress_callback("loading", "Downloading model", True)
            progress_callback("benchmarking", "Running benchmark", True)
            return expected

        with patch.object(advisor, "profile", side_effect=fake_profile):
            items = []
            async for update in advisor.stream_profile():
                items.append(update)

        progress_items = [i for i in items if i.get("phase") != "complete"]
        assert len(progress_items) == 2
        assert progress_items[0]["phase"] == "loading"
        assert progress_items[1]["phase"] == "benchmarking"

    async def test_stream_profile_with_no_progress_events(self):
        expected = _dummy_profiling_result()
        advisor = Advisor("test/model")

        with patch.object(advisor, "profile", return_value=expected):
            items = []
            async for update in advisor.stream_profile():
                items.append(update)

        assert len(items) == 1
        assert items[0]["phase"] == "complete"
        assert items[0]["result"] is expected
