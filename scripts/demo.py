"""
Demo / smoke test script for mlfit Phase 1.

Runs the full recommend pipeline against 4 hardware scenarios and prints
the output to stdout. This is what `mlfit recommend` will look like once
rich + typer are installed.

Usage:
    PYTHONPATH=src python3 scripts/demo.py

No network access required — all hardware/model data is synthetic.
"""
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from mlfit.core.models import ModelProfile, HardwareProfile, GPUInfo, RecommendResult
from mlfit.core.memory import estimate_weights_gb, check_fits
from mlfit.strategies import _load_all_strategies, STRATEGY_REGISTRY

_load_all_strategies()


def build_result(model: ModelProfile, hw: HardwareProfile) -> RecommendResult:
    """Run all compatible strategies and build a ranked RecommendResult."""
    configs = []
    for Cls in STRATEGY_REGISTRY.values():
        s = Cls()
        if s.is_compatible(model, hw):
            configs.append((
                s.estimate_feasibility(model, hw),
                s.generate_config(model, hw),
                s,
            ))
    configs.sort(reverse=True, key=lambda x: x[0].score)

    fits, est_vram, headroom = check_fits(model, hw)
    return RecommendResult(
        hardware=hw, model=model, configs=configs,
        fits=fits, quantized_alternatives=[],
    )


def section(title: str):
    print("\n" + "═" * 70)
    print(f"  {title}")
    print("═" * 70)


def show_result_text(result: RecommendResult):
    """Print a plain-text version of the result (no rich dependency)."""
    hw = result.hardware
    model = result.model

    gpu_str = ", ".join(
        f"{g.name} ({g.vram_free_gb:.1f}GB free)" for g in hw.gpus
    ) or "None"
    avx = "AVX512" if hw.has_avx512 else "AVX2" if hw.has_avx2 else "none"

    print(f"\n  Hardware:")
    print(f"    GPU  : {gpu_str}")
    print(f"    CPU  : {hw.cpu_cores} cores / {hw.cpu_threads} threads · {avx}")
    print(f"    RAM  : {hw.ram_gb:.0f} GB")

    fits_mark = "✓ YES" if result.fits else "✗ NO"
    _, est_vram, headroom = check_fits(model, hw)
    print(f"\n  Model : {model.model_id}")
    print(f"    Architecture : {model.architecture}")
    print(f"    Parameters   : {model.num_parameters / 1e9:.2f}B  ({model.dtype})")
    print(f"    Est. VRAM    : {est_vram:.1f} GB")
    print(f"    Fits         : {fits_mark}  ({headroom:.1f} GB headroom)")

    if result.configs:
        print(f"\n  Ranked configurations:")
        print(f"  {'#':<3} {'Backend':<12} {'Score':>7}  {'TPS':>8}  {'VRAM':>8}  Command")
        print(f"  {'-'*3} {'-'*12} {'-'*7}  {'-'*8}  {'-'*8}  {'-'*30}")
        for i, (score, cfg, _) in enumerate(result.configs, 1):
            cmd_short = cfg.command.split("\n")[0][:50]
            print(f"  {i:<3} {cfg.backend:<12} {score.score:>6.0%}  "
                  f"~{cfg.estimated_tps:>5.0f}t/s  {cfg.estimated_vram_gb:>6.1f}GB  {cmd_short}")

        best = result.configs[0][1]
        print(f"\n  Best command ({best.backend}):")
        for line in best.command.split("\n"):
            print(f"    {line}")
    else:
        print("  [no compatible backends found]")


def show_json_snippet(result: RecommendResult):
    """Show the first config entry from JSON output."""
    import json
    d = result.to_dict()
    if d["configs"]:
        top = d["configs"][0]
        print(f"\n  JSON (top config):")
        print(f"    {json.dumps(top, indent=4)[:500]}")


# ── Scenario 1: NVIDIA RTX 4090 + Llama-3-8B ─────────────────────────────────

section("Scenario 1: NVIDIA RTX 4090 (24GB) + Llama-3-8B")

hw1 = HardwareProfile(
    gpus=[GPUInfo(0, "NVIDIA RTX 4090", 24.0, 22.4, (8, 9), False)],
    total_vram_gb=22.4, gpu_backend="cuda",
    cpu_cores=16, cpu_threads=32, ram_gb=64.0,
    has_avx=True, has_avx2=True, has_avx512=True, disk_free_gb=500.0,
)
model1 = ModelProfile(
    model_id="meta-llama/Llama-3-8B", model_type="llm",
    num_parameters=8.03e9, dtype="bfloat16",
    architecture="LlamaForCausalLM", max_context_length=8192,
    num_hidden_layers=32, num_attention_heads=32,
    num_key_value_heads=8, hidden_size=4096,
)
model1.estimated_size_gb = estimate_weights_gb(model1.num_parameters, model1.dtype)
show_result_text(build_result(model1, hw1))


# ── Scenario 2: Apple M3 Pro + Llama-3-8B ────────────────────────────────────

section("Scenario 2: Apple M3 Pro (36GB unified) + Llama-3-8B")

hw2 = HardwareProfile(
    gpus=[GPUInfo(0, "Apple M3 Pro", 36.0, 28.0, (0, 0), True)],
    total_vram_gb=28.0, gpu_backend="mps",
    cpu_cores=12, cpu_threads=12, ram_gb=36.0,
    has_avx=False, has_avx2=False, has_avx512=False, disk_free_gb=400.0,
)
show_result_text(build_result(model1, hw2))


# ── Scenario 3: CPU-only server + Qwen 0.5B ──────────────────────────────────

section("Scenario 3: CPU-only server (256GB RAM) + Qwen 0.5B")

hw3 = HardwareProfile(
    gpus=[], total_vram_gb=0.0, gpu_backend="none",
    cpu_cores=32, cpu_threads=64, ram_gb=256.0,
    has_avx=True, has_avx2=True, has_avx512=True, disk_free_gb=2000.0,
)
model3 = ModelProfile(
    model_id="Qwen/Qwen2.5-0.5B", model_type="llm",
    num_parameters=0.5e9, dtype="bfloat16",
    architecture="Qwen2ForCausalLM", max_context_length=32768,
    num_hidden_layers=24, num_attention_heads=14,
    num_key_value_heads=2, hidden_size=896,
)
model3.estimated_size_gb = estimate_weights_gb(model3.num_parameters, model3.dtype)
show_result_text(build_result(model3, hw3))


# ── Scenario 4: Small GPU + Large model (doesn't fit) ────────────────────────

section("Scenario 4: RTX 3080 10GB + Llama-3-70B (doesn't fit)")

hw4 = HardwareProfile(
    gpus=[GPUInfo(0, "NVIDIA RTX 3080 10GB", 10.0, 9.0, (8, 6), False)],
    total_vram_gb=9.0, gpu_backend="cuda",
    cpu_cores=8, cpu_threads=16, ram_gb=32.0,
    has_avx=True, has_avx2=True, has_avx512=False, disk_free_gb=200.0,
)
model4 = ModelProfile(
    model_id="meta-llama/Llama-3-70B", model_type="llm",
    num_parameters=70e9, dtype="bfloat16",
    architecture="LlamaForCausalLM", max_context_length=8192,
    num_hidden_layers=80, num_attention_heads=64,
    num_key_value_heads=8, hidden_size=8192,
)
model4.estimated_size_gb = estimate_weights_gb(model4.num_parameters, model4.dtype)
show_result_text(build_result(model4, hw4))


# ── JSON output demo ──────────────────────────────────────────────────────────

section("JSON output (--output json) — Scenario 1")
show_json_snippet(build_result(model1, hw1))

print("\n" + "═" * 70)
print("  Demo complete. All 4 scenarios passed.")
print("═" * 70 + "\n")
