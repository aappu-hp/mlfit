# mlfit

**Universal Model Deployment Advisor** — find the best way to serve any model on your hardware in seconds.

mlfit analyzes your hardware and a HuggingFace model, scores every compatible backend, and hands you a ranked deployment plan with ready-to-run commands.

---

## What it does

1. **Detects your hardware** — VRAM, RAM, CPU threads, GPU backend (CUDA / MPS / CPU)
2. **Analyzes the model** — parameters, dtype, architecture, quantization type, context length
3. **Scores all backends** — vLLM, TGI, Ollama, llama.cpp, ONNX Runtime, scikit-learn
4. **Recommends the best fit** — ranked list with estimated TPS, VRAM usage, and exact commands
5. **Profiles live** — optional dynamic benchmarking against a real running server

---

## Installation

Requires Python 3.10+ and [uv](https://docs.astral.sh/uv/).

```bash
git clone https://github.com/your-org/mlfit
cd mlfit
uv sync
```

Optional extras:

```bash
uv sync --extra gpu    # torch for CUDA detection
uv sync --extra gguf   # gguf parsing for GGUF models
uv sync --extra onnx   # onnxruntime for ONNX models
```

---

## CLI usage

### recommend

Get a ranked deployment plan for any HuggingFace model:

```bash
mlfit recommend meta-llama/Llama-3-8B
mlfit recommend meta-llama/Llama-3-8B --backend vllm
mlfit recommend meta-llama/Llama-3-8B --json
```

### compare

Compare all compatible backends side-by-side in a Rich table:

```bash
mlfit compare meta-llama/Llama-3-8B
```

Shows estimated TPS, VRAM, setup complexity, quantization support, and a winner-by-use-case summary.

### profile

Dynamically benchmark a running inference server:

```bash
mlfit profile meta-llama/Llama-3-8B --backend vllm --concurrency 1,4,16
```

Outputs real TPS, TTFT p50/p95, and peak VRAM measurements.

---

## Python library

mlfit is also a proper Python library. Import it directly in your code:

```python
from mlfit import recommend, Advisor, detect_hardware

# one-shot convenience function
result = recommend("meta-llama/Llama-3-8B")
print(result.best.command)         # ready-to-run deploy command
print(result.best.estimated_tps)   # estimated tokens/sec
print(result.to_json())            # full JSON output

# class-based for stateful reuse
advisor = Advisor("meta-llama/Llama-3-8B")
result = advisor.recommend()
for cfg in result.alternatives:
    print(cfg.backend, cfg.estimated_tps)

# inspect hardware independently
hw = detect_hardware()
print(hw.gpu_backend, hw.total_vram_gb)
```

### Async support

```python
import asyncio
from mlfit import Advisor

async def main():
    advisor = Advisor("meta-llama/Llama-3-8B")

    # non-blocking — safe inside FastAPI, async scripts, Jupyter
    result = await advisor.profile_async(backend="vllm", concurrency=[1, 4, 16])
    print(result.tps_at_concurrency(4))

asyncio.run(main())
```

### Streaming profiling

```python
async def stream():
    advisor = Advisor("meta-llama/Llama-3-8B")
    async for update in advisor.stream_profile(backend="vllm"):
        if update.get("result"):
            bench = update["result"]
        else:
            print(update["phase"], "—", update["message"])
```

---

## Supported backends

| Backend | Best for | GPU required |
|---|---|---|
| **vLLM** | High-throughput LLM serving, PagedAttention | CUDA |
| **TGI** | HuggingFace-native serving, Inference Endpoints | CUDA |
| **Ollama** | Local interactive use, easy setup | Optional |
| **llama.cpp** | CPU edge, Apple Silicon, GGUF models | Optional |
| **ONNX Runtime** | Optimized inference, cross-platform | Optional |
| **scikit-learn** | Classical ML models (joblib) | CPU only |

---

## Supported hardware

| Hardware | GPU backend | Notes |
|---|---|---|
| NVIDIA (CUDA) | `cuda` | Full backend support including vLLM + TGI |
| Apple Silicon | `mps` | Ollama + llama.cpp with MPS acceleration |
| CPU only | `none` | llama.cpp + ONNX Runtime + sklearn |
| AMD (ROCm) | `rocm` | Detected, partial backend support |

---

## Development

```bash
uv sync --group dev
uv run pytest
uv run pytest tests/unit/test_strategies.py -v
```
