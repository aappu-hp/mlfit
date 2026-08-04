import subprocess
import threading
import time
from unittest.mock import MagicMock, patch, PropertyMock

import pytest

from mlfit.core.models import BenchmarkPoint, ProfilingResult
from mlfit.profiler.latency import LatencyResult, _measure_one_ttft
from mlfit.profiler.memory import MemorySnapshot, MemoryTracker
from mlfit.profiler.server import ServerManager
from mlfit.profiler.throughput import ThroughputResult
from mlfit.strategies.base import BackendConfig


# ── fixtures ────────────────────────────────────────────────────────────────

@pytest.fixture
def dummy_config():
    return BackendConfig(
        backend="vllm",
        model_id="test/model",
        params={"gpu_memory_utilization": 0.85, "max_model_len": 4096},
        command="vllm serve test/model --gpu-memory-utilization 0.85",
        estimated_vram_gb=10.0,
        estimated_tps=50.0,
        server_url="http://localhost:8000",
        health_path="/health",
    )


@pytest.fixture
def ollama_config():
    return BackendConfig(
        backend="ollama",
        model_id="test/model",
        params={"num_ctx": 4096},
        command="ollama run test-model",
        estimated_vram_gb=5.0,
        estimated_tps=40.0,
        server_url="http://localhost:11434",
        server_command="ollama serve",
        health_path="/api/tags",
    )


# ── MemoryTracker ────────────────────────────────────────────────────────────

class TestMemoryTracker:
    def test_peak_vram_returns_maximum_sample(self, mocker):
        mocker.patch("psutil.virtual_memory", return_value=MagicMock(
            total=64 * 1e9, available=48 * 1e9,
        ))
        tracker = MemoryTracker(poll_interval_s=0.01)
        tracker._snapshots = [
            MemorySnapshot(vram_gb=5.0, ram_gb=10.0),
            MemorySnapshot(vram_gb=12.3, ram_gb=11.0),
            MemorySnapshot(vram_gb=8.1, ram_gb=10.5),
        ]
        assert tracker.peak_vram_gb() == pytest.approx(12.3)

    def test_peak_vram_returns_zero_when_no_samples(self):
        tracker = MemoryTracker()
        assert tracker.peak_vram_gb() == 0.0

    def test_peak_ram_returns_maximum_sample(self):
        tracker = MemoryTracker()
        tracker._snapshots = [
            MemorySnapshot(vram_gb=1.0, ram_gb=20.0),
            MemorySnapshot(vram_gb=1.0, ram_gb=35.5),
        ]
        assert tracker.peak_ram_gb() == pytest.approx(35.5)

    def test_context_manager_starts_and_stops_thread(self, mocker):
        mocker.patch(
            "mlfit.profiler.memory.MemoryTracker._take_snapshot",
            return_value=MemorySnapshot(vram_gb=2.0, ram_gb=8.0),
        )
        with MemoryTracker(poll_interval_s=0.01) as tracker:
            assert tracker._thread is not None
            time.sleep(0.05)
        assert tracker._stop.is_set()

    def test_read_vram_falls_back_to_psutil_on_apple_silicon(self, mocker):
        # Simulate no pynvml and no torch — psutil fallback should produce a non-negative value
        mocker.patch("mlfit.profiler.memory.MemoryTracker._read_vram_gb", return_value=16.0)
        tracker = MemoryTracker()
        assert tracker._read_vram_gb() >= 0.0


# ── LatencyResult ───────────────────────────────────────────────────────────

class TestLatencyResult:
    def test_p50_is_median(self):
        result = LatencyResult(samples_ms=[100.0, 200.0, 300.0])
        assert result.p50_ms == pytest.approx(200.0)

    def test_p95_selects_correct_index(self):
        samples = list(range(1, 21))  # 1..20 ms
        result = LatencyResult(samples_ms=[float(s) for s in samples])
        # index = int(20 * 0.95) = 19 → sorted[19] = 20.0
        assert result.p95_ms == pytest.approx(20.0)

    def test_p99_selects_correct_index(self):
        samples = list(range(1, 101))  # 1..100 ms
        result = LatencyResult(samples_ms=[float(s) for s in samples])
        assert result.p99_ms == pytest.approx(100.0)

    def test_percentiles_return_zero_for_empty_samples(self):
        result = LatencyResult(samples_ms=[])
        assert result.p50_ms == 0.0
        assert result.p95_ms == 0.0
        assert result.p99_ms == 0.0

    def test_single_sample_returns_that_value_for_all_percentiles(self):
        result = LatencyResult(samples_ms=[42.0])
        assert result.p50_ms == pytest.approx(42.0)
        assert result.p95_ms == pytest.approx(42.0)
        assert result.p99_ms == pytest.approx(42.0)


# ── ThroughputResult ─────────────────────────────────────────────────────────

class TestThroughputResult:
    def test_fields_are_stored_correctly(self):
        result = ThroughputResult(
            concurrency=4,
            tps=123.4,
            total_tokens=500,
            elapsed_s=4.05,
        )
        assert result.concurrency == 4
        assert result.tps == pytest.approx(123.4)
        assert result.total_tokens == 500


# ── ServerManager ────────────────────────────────────────────────────────────

class TestServerManager:
    def test_start_launches_subprocess_with_config_command(self, mocker, dummy_config):
        mock_popen = mocker.patch("subprocess.Popen")
        manager = ServerManager(dummy_config)
        manager.start()
        mock_popen.assert_called_once()
        cmd_arg = mock_popen.call_args[0][0]
        assert "vllm" in cmd_arg[0]

    def test_start_uses_server_command_when_set(self, mocker, ollama_config):
        mock_popen = mocker.patch("subprocess.Popen")
        manager = ServerManager(ollama_config)
        manager.start()
        cmd_arg = mock_popen.call_args[0][0]
        assert cmd_arg[0] == "ollama"
        assert cmd_arg[1] == "serve"

    def test_base_url_returns_config_server_url(self, dummy_config):
        manager = ServerManager(dummy_config)
        assert manager.base_url == "http://localhost:8000"

    def test_stop_terminates_running_process(self, mocker, dummy_config):
        mock_process = MagicMock()
        mock_process.poll.return_value = None
        mock_process.pid = 1234
        mocker.patch("subprocess.Popen", return_value=mock_process)
        manager = ServerManager(dummy_config)
        manager.start()
        manager.stop()
        mock_process.terminate.assert_called_once()

    def test_wait_until_ready_returns_true_on_200(self, mocker, dummy_config):
        mocker.patch("subprocess.Popen")
        mocker.patch(
            "mlfit.profiler.server.ServerManager._has_crashed",
            return_value=False,
        )
        mock_get = mocker.patch("httpx.get")
        mock_get.return_value = MagicMock(status_code=200)

        manager = ServerManager(dummy_config, timeout_s=10)
        manager._process = MagicMock()
        manager._process.poll.return_value = None

        assert manager.wait_until_ready() is True

    def test_wait_until_ready_returns_false_when_process_crashes(
        self, mocker, dummy_config
    ):
        mocker.patch("mlfit.profiler.server._POLL_INTERVAL_S", 0.01)
        mocker.patch(
            "mlfit.profiler.server.ServerManager._has_crashed",
            return_value=True,
        )
        manager = ServerManager(dummy_config, timeout_s=5)
        manager._process = MagicMock()
        manager._process.returncode = 1

        assert manager.wait_until_ready() is False

    def test_context_manager_calls_start_and_stop(self, mocker, dummy_config):
        mock_start = mocker.patch.object(ServerManager, "start")
        mock_stop = mocker.patch.object(ServerManager, "stop")
        with ServerManager(dummy_config):
            pass
        mock_start.assert_called_once()
        mock_stop.assert_called_once()

    def test_is_running_false_before_start(self, dummy_config):
        manager = ServerManager(dummy_config)
        assert manager.is_running() is False


# ── BenchmarkPoint and ProfilingResult ──────────────────────────────────────

class TestBenchmarkPoint:
    def test_stores_all_fields(self):
        bp = BenchmarkPoint(
            concurrency=4,
            tps=198.2,
            ttft_p50_ms=142.0,
            ttft_p95_ms=318.0,
        )
        assert bp.concurrency == 4
        assert bp.tps == pytest.approx(198.2)
        assert bp.ttft_p50_ms == pytest.approx(142.0)
        assert bp.ttft_p95_ms == pytest.approx(318.0)


class TestProfilingResult:
    def _make_result(self, dummy_config, mocker):
        from mlfit.core.models import HardwareProfile
        hw = HardwareProfile(
            gpus=[], total_vram_gb=0.0, gpu_backend="none",
            cpu_cores=8, cpu_threads=16, ram_gb=32.0,
            has_avx=True, has_avx2=True, has_avx512=False,
            disk_free_gb=100.0,
        )
        return ProfilingResult(
            model_id="test/model",
            backend="vllm",
            hardware=hw,
            final_config=dummy_config,
            peak_vram_gb=19.4,
            benchmark_points=[
                BenchmarkPoint(concurrency=1, tps=68.4, ttft_p50_ms=142.0, ttft_p95_ms=318.0),
                BenchmarkPoint(concurrency=4, tps=198.2, ttft_p50_ms=142.0, ttft_p95_ms=318.0),
            ],
            time_elapsed_s=372.0,
            timestamp="2026-07-08T10:00:00Z",
        )

    def test_to_dict_contains_required_keys(self, dummy_config, mocker):
        result = self._make_result(dummy_config, mocker)
        d = result.to_dict()
        assert "model_id" in d
        assert "backend" in d
        assert "memory" in d
        assert "benchmarks" in d
        assert len(d["benchmarks"]) == 2

    def test_to_dict_benchmark_concurrency_values(self, dummy_config, mocker):
        result = self._make_result(dummy_config, mocker)
        concurrencies = [b["concurrency"] for b in result.to_dict()["benchmarks"]]
        assert concurrencies == [1, 4]

    def test_to_json_is_valid_json(self, dummy_config, mocker):
        import json
        result = self._make_result(dummy_config, mocker)
        parsed = json.loads(result.to_json())
        assert parsed["model_id"] == "test/model"

    def test_to_dict_peak_vram_rounded(self, dummy_config, mocker):
        result = self._make_result(dummy_config, mocker)
        assert result.to_dict()["memory"]["peak_vram_gb"] == pytest.approx(19.4)


# ── generate_config_candidates ───────────────────────────────────────────────

class TestVLLMConfigCandidates:
    def test_returns_four_candidates_in_ascending_mem_util(self):
        from mlfit.strategies.vllm import VLLMStrategy
        from tests.unit.test_strategies import make_llama_8b, rtx_4090

        strategy = VLLMStrategy()
        candidates = strategy.generate_config_candidates(make_llama_8b(), rtx_4090())

        assert len(candidates) == 4
        utils = [c.params["gpu_memory_utilization"] for c in candidates]
        assert utils == sorted(utils), "Candidates must be ordered least to most aggressive"

    def test_all_candidates_have_server_url(self):
        from mlfit.strategies.vllm import VLLMStrategy
        from tests.unit.test_strategies import make_llama_8b, rtx_4090

        strategy = VLLMStrategy()
        candidates = strategy.generate_config_candidates(make_llama_8b(), rtx_4090())
        for cfg in candidates:
            assert cfg.server_url == "http://localhost:8000"


class TestOllamaConfigCandidates:
    def test_returns_single_candidate(self):
        from mlfit.strategies.ollama import OllamaStrategy
        from tests.unit.test_strategies import make_llama_8b, macbook_m3

        strategy = OllamaStrategy()
        candidates = strategy.generate_config_candidates(make_llama_8b(), macbook_m3())
        assert len(candidates) == 1

    def test_candidate_has_correct_server_fields(self):
        from mlfit.strategies.ollama import OllamaStrategy
        from tests.unit.test_strategies import make_llama_8b, macbook_m3

        strategy = OllamaStrategy()
        candidates = strategy.generate_config_candidates(make_llama_8b(), macbook_m3())
        cfg = candidates[0]
        assert cfg.server_url == "http://localhost:11434"
        assert cfg.server_command == "ollama serve"
        assert cfg.health_path == "/api/tags"
