"""Unit tests for backend strategies — no GPU or network required."""
import unittest
from mlfit.core.models import ModelProfile, HardwareProfile, GPUInfo
from mlfit.core.memory import estimate_weights_gb


# ── Fixtures ──────────────────────────────────────────────────────────────────

def make_llama_8b() -> ModelProfile:
    m = ModelProfile(
        model_id="meta-llama/Llama-3-8B",
        model_type="llm",
        num_parameters=8.03e9,
        dtype="bfloat16",
        architecture="LlamaForCausalLM",
        max_context_length=8192,
        num_hidden_layers=32,
        num_attention_heads=32,
        num_key_value_heads=8,
        hidden_size=4096,
    )
    m.estimated_size_gb = estimate_weights_gb(m.num_parameters, m.dtype)
    return m


def make_gguf_7b() -> ModelProfile:
    m = ModelProfile(
        model_id="Qwen/Qwen2.5-7B-GGUF:Q4_K_M",
        model_type="gguf",
        num_parameters=7e9,
        dtype="int4",
        architecture="Qwen2ForCausalLM",
        max_context_length=32768,
        num_hidden_layers=28,
        num_attention_heads=28,
        num_key_value_heads=4,
        hidden_size=3584,
        quant_type="Q4_K_M",
    )
    m.estimated_size_gb = 4.1
    return m


def rtx_4090() -> HardwareProfile:
    return HardwareProfile(
        gpus=[GPUInfo(0, "NVIDIA RTX 4090", 24.0, 22.4, (8, 9), False)],
        total_vram_gb=22.4, gpu_backend="cuda",
        cpu_cores=16, cpu_threads=32, ram_gb=64.0,
        has_avx=True, has_avx2=True, has_avx512=True, disk_free_gb=500.0,
    )


def macbook_m3() -> HardwareProfile:
    return HardwareProfile(
        gpus=[GPUInfo(0, "Apple M3 Pro", 36.0, 28.0, (0, 0), True)],
        total_vram_gb=28.0, gpu_backend="mps",
        cpu_cores=12, cpu_threads=12, ram_gb=36.0,
        has_avx=False, has_avx2=False, has_avx512=False, disk_free_gb=400.0,
    )


def cpu_server() -> HardwareProfile:
    return HardwareProfile(
        gpus=[], total_vram_gb=0.0, gpu_backend="none",
        cpu_cores=32, cpu_threads=64, ram_gb=256.0,
        has_avx=True, has_avx2=True, has_avx512=True, disk_free_gb=2000.0,
    )


# ── vLLM Strategy Tests ───────────────────────────────────────────────────────

class TestVLLMStrategy(unittest.TestCase):
    def setUp(self):
        from mlfit.strategies.vllm import VLLMStrategy
        self.strategy = VLLMStrategy()

    def test_compatible_with_cuda_llm(self):
        assert self.strategy.is_compatible(make_llama_8b(), rtx_4090())

    def test_not_compatible_with_mps(self):
        assert not self.strategy.is_compatible(make_llama_8b(), macbook_m3())

    def test_not_compatible_with_cpu_only(self):
        assert not self.strategy.is_compatible(make_llama_8b(), cpu_server())

    def test_high_feasibility_when_model_fits_well(self):
        score = self.strategy.estimate_feasibility(make_llama_8b(), rtx_4090())
        assert score.score >= 0.70, f"Expected high score, got {score.score}"

    def test_generates_valid_config(self):
        config = self.strategy.generate_config(make_llama_8b(), rtx_4090())
        assert config.backend == "vllm"
        assert "gpu_memory_utilization" in config.params
        assert 0.5 <= config.params["gpu_memory_utilization"] <= 1.0
        assert "max_model_len" in config.params
        assert "vllm serve" in config.command

    def test_enforces_eager_when_tight(self):
        """When model barely fits, enforce_eager should be True."""
        tight_hw = HardwareProfile(
            gpus=[GPUInfo(0, "RTX 3080", 10.0, 9.5, (8, 6), False)],
            total_vram_gb=9.5, gpu_backend="cuda",
            cpu_cores=8, cpu_threads=16, ram_gb=32.0,
            has_avx=True, has_avx2=True, has_avx512=False, disk_free_gb=100.0,
        )
        # 8B model (16GB) vs 9.5GB VRAM — very tight
        config = self.strategy.generate_config(make_llama_8b(), tight_hw)
        assert config.params.get("enforce_eager", False), "Should use enforce_eager for tight fit"


# ── Ollama Strategy Tests ─────────────────────────────────────────────────────

class TestOllamaStrategy(unittest.TestCase):
    def setUp(self):
        from mlfit.strategies.ollama import OllamaStrategy
        self.strategy = OllamaStrategy()

    def test_compatible_with_mps(self):
        assert self.strategy.is_compatible(make_llama_8b(), macbook_m3())

    def test_compatible_with_cuda(self):
        assert self.strategy.is_compatible(make_llama_8b(), rtx_4090())

    def test_compatible_with_cpu(self):
        assert self.strategy.is_compatible(make_llama_8b(), cpu_server())

    def test_high_score_on_mps(self):
        score = self.strategy.estimate_feasibility(make_llama_8b(), macbook_m3())
        assert score.score >= 0.85, f"Ollama should score very high on MPS, got {score.score}"

    def test_generates_ollama_run_command(self):
        config = self.strategy.generate_config(make_llama_8b(), rtx_4090())
        assert "ollama run" in config.command
        assert config.backend == "ollama"


# ── llama.cpp Strategy Tests ──────────────────────────────────────────────────

class TestLlamaCppStrategy(unittest.TestCase):
    def setUp(self):
        from mlfit.strategies.llamacpp import LlamaCppStrategy
        self.strategy = LlamaCppStrategy()

    def test_compatible_everywhere(self):
        for hw in [rtx_4090(), macbook_m3(), cpu_server()]:
            assert self.strategy.is_compatible(make_llama_8b(), hw)

    def test_compatible_with_gguf(self):
        assert self.strategy.is_compatible(make_gguf_7b(), cpu_server())

    def test_generates_llama_server_command(self):
        config = self.strategy.generate_config(make_gguf_7b(), cpu_server())
        assert "llama-server" in config.command
        assert "--host 0.0.0.0" in config.command

    def test_zero_vram_on_cpu(self):
        config = self.strategy.generate_config(make_llama_8b(), cpu_server())
        assert config.estimated_vram_gb == 0.0

    def test_gpu_layers_set_for_cuda(self):
        config = self.strategy.generate_config(make_llama_8b(), rtx_4090())
        assert config.params["n_gpu_layers"] == 999


# ── TGI Strategy Tests ────────────────────────────────────────────────────────

class TestTGIStrategy(unittest.TestCase):
    def setUp(self):
        from mlfit.strategies.tgi import TGIStrategy
        self.strategy = TGIStrategy()

    def test_requires_cuda(self):
        assert self.strategy.is_compatible(make_llama_8b(), rtx_4090())
        assert not self.strategy.is_compatible(make_llama_8b(), macbook_m3())
        assert not self.strategy.is_compatible(make_llama_8b(), cpu_server())

    def test_generates_docker_command(self):
        config = self.strategy.generate_config(make_llama_8b(), rtx_4090())
        assert "docker run" in config.command
        assert "text-generation-inference" in config.command

    def test_adds_quantize_flag_for_awq_model(self):
        model = make_llama_8b()
        model.quant_type = "AWQ"
        config = self.strategy.generate_config(model, rtx_4090())
        assert config.params.get("quantize") == "awq"
        assert "--quantize awq" in config.command

    def test_adds_quantize_flag_for_gptq_model(self):
        model = make_llama_8b()
        model.quant_type = "GPTQ"
        config = self.strategy.generate_config(model, rtx_4090())
        assert config.params.get("quantize") == "gptq"
        assert "--quantize gptq" in config.command

    def test_no_quantize_flag_for_comfortable_fit(self):
        config = self.strategy.generate_config(make_llama_8b(), rtx_4090())
        assert "quantize" not in config.params


# ── ONNX Runtime Strategy Tests ───────────────────────────────────────────────

def make_onnx_model() -> "ModelProfile":
    m = ModelProfile(
        model_id="./model.onnx",
        model_type="onnx",
        num_parameters=0,
        dtype="float32",
        architecture="ONNX",
        max_context_length=0,
        num_hidden_layers=0,
        num_attention_heads=0,
        num_key_value_heads=0,
        hidden_size=0,
    )
    m.estimated_size_gb = 0.25
    return m


class TestONNXRuntimeStrategy(unittest.TestCase):
    def setUp(self):
        from mlfit.strategies.onnxruntime import ONNXRuntimeStrategy
        self.strategy = ONNXRuntimeStrategy()

    def test_compatible_with_onnx_only(self):
        assert self.strategy.is_compatible(make_onnx_model(), cpu_server())

    def test_not_compatible_with_llm(self):
        assert not self.strategy.is_compatible(make_llama_8b(), rtx_4090())

    def test_not_compatible_with_sklearn(self):
        sklearn_model = ModelProfile(
            model_id="sklearn:RandomForestClassifier",
            model_type="sklearn",
            num_parameters=0, dtype="float64",
            architecture="RandomForestClassifier",
            max_context_length=0, num_hidden_layers=0,
            num_attention_heads=0, num_key_value_heads=0, hidden_size=0,
        )
        assert not self.strategy.is_compatible(sklearn_model, cpu_server())

    def test_cuda_scores_highest(self):
        cpu_score = self.strategy.estimate_feasibility(make_onnx_model(), cpu_server()).score
        cuda_score = self.strategy.estimate_feasibility(make_onnx_model(), rtx_4090()).score
        assert cuda_score > cpu_score

    def test_avx512_scores_higher_than_avx2(self):
        avx2_hw = HardwareProfile(
            gpus=[], total_vram_gb=0.0, gpu_backend="none",
            cpu_cores=16, cpu_threads=32, ram_gb=64.0,
            has_avx=True, has_avx2=True, has_avx512=False, disk_free_gb=500.0,
        )
        avx512_score = self.strategy.estimate_feasibility(make_onnx_model(), cpu_server()).score
        avx2_score = self.strategy.estimate_feasibility(make_onnx_model(), avx2_hw).score
        assert avx512_score > avx2_score

    def test_cuda_providers_on_cuda_hw(self):
        config = self.strategy.generate_config(make_onnx_model(), rtx_4090())
        assert "CUDAExecutionProvider" in config.params["providers"]

    def test_cpu_provider_on_cpu_hw(self):
        config = self.strategy.generate_config(make_onnx_model(), cpu_server())
        assert config.params["providers"] == ["CPUExecutionProvider"]

    def test_coreml_provider_on_mps(self):
        config = self.strategy.generate_config(make_onnx_model(), macbook_m3())
        assert "CoreMLExecutionProvider" in config.params["providers"]

    def test_intra_threads_set_to_cpu_cores(self):
        config = self.strategy.generate_config(make_onnx_model(), cpu_server())
        assert config.params["intra_op_num_threads"] == cpu_server().cpu_cores

    def test_command_is_python_snippet(self):
        config = self.strategy.generate_config(make_onnx_model(), cpu_server())
        assert "ort.InferenceSession" in config.command
        assert "ort.SessionOptions" in config.command

    def test_zero_vram_on_cpu(self):
        config = self.strategy.generate_config(make_onnx_model(), cpu_server())
        assert config.estimated_vram_gb == 0.0

    def test_format_key_settings(self):
        config = self.strategy.generate_config(make_onnx_model(), cpu_server())
        key = self.strategy.format_key_settings(config.params)
        assert "provider" in key
        assert "threads" in key


# ── sklearn Strategy Tests ────────────────────────────────────────────────────

def make_sklearn_model(class_name: str = "RandomForestClassifier") -> "ModelProfile":
    m = ModelProfile(
        model_id=f"sklearn:{class_name}",
        model_type="sklearn",
        num_parameters=0,
        dtype="float64",
        architecture=class_name,
        max_context_length=0,
        num_hidden_layers=0,
        num_attention_heads=0,
        num_key_value_heads=0,
        hidden_size=0,
    )
    m.estimated_size_gb = 0.001
    return m


class TestSklearnStrategy(unittest.TestCase):
    def setUp(self):
        from mlfit.strategies.sklearn_strategy import SklearnStrategy
        self.strategy = SklearnStrategy()

    def test_compatible_with_sklearn_only(self):
        assert self.strategy.is_compatible(make_sklearn_model(), cpu_server())

    def test_not_compatible_with_llm(self):
        assert not self.strategy.is_compatible(make_llama_8b(), cpu_server())

    def test_not_compatible_with_onnx(self):
        assert not self.strategy.is_compatible(make_onnx_model(), cpu_server())

    def test_high_feasibility_for_ensemble(self):
        score = self.strategy.estimate_feasibility(make_sklearn_model("RandomForestClassifier"), cpu_server())
        assert score.score >= 0.90

    def test_lower_feasibility_for_mlp(self):
        ensemble_score = self.strategy.estimate_feasibility(
            make_sklearn_model("RandomForestClassifier"), cpu_server()
        ).score
        mlp_score = self.strategy.estimate_feasibility(
            make_sklearn_model("MLPClassifier"), cpu_server()
        ).score
        assert ensemble_score > mlp_score

    def test_n_jobs_matches_cpu_threads(self):
        config = self.strategy.generate_config(make_sklearn_model(), cpu_server())
        assert config.params["n_jobs"] == cpu_server().cpu_threads

    def test_zero_vram(self):
        config = self.strategy.generate_config(make_sklearn_model(), cpu_server())
        assert config.estimated_vram_gb == 0.0

    def test_command_is_python_snippet(self):
        config = self.strategy.generate_config(make_sklearn_model(), cpu_server())
        assert "joblib" in config.command
        assert "model.predict" in config.command

    def test_format_key_settings(self):
        config = self.strategy.generate_config(make_sklearn_model(), cpu_server())
        key = self.strategy.format_key_settings(config.params)
        assert "n_jobs" in key

    def test_linear_model_throughput_higher_than_ensemble(self):
        forest_config = self.strategy.generate_config(
            make_sklearn_model("RandomForestClassifier"), cpu_server()
        )
        linear_config = self.strategy.generate_config(
            make_sklearn_model("LogisticRegression"), cpu_server()
        )
        assert linear_config.estimated_tps > forest_config.estimated_tps


# ── llama.cpp Phase 4 Improvements ───────────────────────────────────────────

class TestLlamaCppPhase4(unittest.TestCase):
    def setUp(self):
        from mlfit.strategies.llamacpp import LlamaCppStrategy
        self.strategy = LlamaCppStrategy()

    def test_mlock_included_on_cpu_only(self):
        config = self.strategy.generate_config(make_llama_8b(), cpu_server())
        assert config.params.get("mlock") is True
        assert "--mlock" in config.command

    def test_mlock_not_included_on_gpu(self):
        config = self.strategy.generate_config(make_llama_8b(), rtx_4090())
        assert "mlock" not in config.params

    def test_larger_ctx_on_gpu(self):
        gpu_config = self.strategy.generate_config(make_llama_8b(), rtx_4090())
        cpu_config = self.strategy.generate_config(make_llama_8b(), cpu_server())
        assert gpu_config.params["ctx_size"] > cpu_config.params["ctx_size"]

    def test_threads_batch_on_mps(self):
        config = self.strategy.generate_config(make_llama_8b(), macbook_m3())
        assert "threads_batch" in config.params
        assert "--tb" in config.command

    def test_no_threads_batch_on_cuda(self):
        config = self.strategy.generate_config(make_llama_8b(), rtx_4090())
        assert "threads_batch" not in config.params
