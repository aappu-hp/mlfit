from mlfit.detectors.runtimes import detect_runtimes

_EXPECTED_RUNTIMES = {"Ollama", "llama.cpp", "vLLM", "MLX", "LM Studio"}


def test_detect_runtimes_returns_expected_keys():
    result = detect_runtimes()
    assert set(result.keys()) == _EXPECTED_RUNTIMES


def test_detect_runtimes_values_are_bool():
    result = detect_runtimes()
    assert all(isinstance(available, bool) for available in result.values())


def test_detect_runtimes_honors_llama_cpp_path_env(monkeypatch):
    monkeypatch.setenv("LLAMA_CPP_PATH", "/some/where/llama")
    assert detect_runtimes()["llama.cpp"] is True
