import typer
from rich.console import Console
from rich.table import Table
from rich import box

console = Console()

_QUANT_SUPPORT: dict[str, str] = {
    "vllm": "AWQ, GPTQ, FP8",
    "tgi": "GPTQ, AWQ, BnB",
    "ollama": "GGUF only",
    "llamacpp": "GGUF only",
    "onnxruntime": "N/A",
    "sklearn": "N/A",
}

_SETUP_COMPLEXITY: dict[str, str] = {
    "vllm": "Medium",
    "tgi": "Hard (Docker)",
    "ollama": "Easy",
    "llamacpp": "Medium",
    "onnxruntime": "Easy",
    "sklearn": "Easy",
}

_DOCKER_REQUIRED: dict[str, str] = {
    "vllm": "No",
    "tgi": "Yes",
    "ollama": "No",
    "llamacpp": "No",
    "onnxruntime": "No",
    "sklearn": "No",
}


def run(
    model_id: str = typer.Argument(...),
    backends: str = typer.Option("vllm,ollama,llamacpp", "--backends",
        help="Comma-separated backends to compare: vllm,ollama,llamacpp,tgi"),
    gpuid: str = typer.Option("all", "--gpuid", "-g"),
    output: str = typer.Option("table", "--output", "-o"),
):
    """
    Compare multiple backends side-by-side for a model.

    Example:
      mlfit compare Qwen/Qwen2.5-7B --backends vllm,ollama,llamacpp
    """
    import json
    from mlfit.core.advisor import Advisor
    from mlfit.reporter.table import render_header, render_hardware_panel

    backend_list = [b.strip() for b in backends.split(",")]

    advisor = Advisor(model_id=model_id, gpuid=gpuid)
    hardware = advisor.hardware
    model = advisor.model

    render_header()
    render_hardware_panel(hardware)

    console.print(f"\n  Comparing [bold]{len(backend_list)}[/bold] backends for [cyan]{model_id}[/cyan]...\n")

    results = {}
    for backend in backend_list:
        try:
            result = advisor.recommend(backend=backend)
            if result.configs:
                results[backend] = result.configs[0]  # (score, config, strategy)
        except Exception as e:
            console.print(f"[yellow]⚠  {backend}: {e}[/yellow]")

    if not results:
        console.print("[red]No backends produced results.[/red]")
        raise typer.Exit(1)

    if output == "json":
        out = {
            "model_id": model_id,
            "backends": {
                b: {
                    "feasibility_score": round(r[0].score, 2),
                    "estimated_tps": r[1].estimated_tps,
                    "estimated_vram_gb": r[1].estimated_vram_gb,
                    "command": r[1].command,
                    "params": r[1].params,
                }
                for b, r in results.items()
            }
        }
        typer.echo(json.dumps(out, indent=2))
        return

    _render_comparison_table(model_id, model, results)
    _render_winner_section(results)
    _render_commands_section(results)


def _render_comparison_table(model_id: str, model, results: dict) -> None:
    """
    Render the side-by-side property comparison table.

    Rows cover feasibility, throughput, memory, concurrency, API compatibility,
    quantization support, multi-GPU capability, setup complexity, and Docker
    requirements.

    Args:
        model_id: Model identifier shown in the table title.
        model: ModelProfile (currently unused but available for future rows).
        results: Dict mapping backend name → (FeasibilityScore, BackendConfig, strategy).
    """
    table = Table(
        title=f"Side-by-Side Comparison: {model_id}",
        box=box.ROUNDED,
        show_header=True,
    )

    table.add_column("Property", style="bold", width=28)
    for backend in results:
        table.add_column(backend, style="cyan", justify="center")

    def row(label: str, fn):
        table.add_row(label, *[fn(results[b]) for b in results])

    row("Feasibility",
        lambda r: f"{int(r[0].score * 100)}%")
    row("Est. TPS (batch=1)",
        lambda r: f"~{r[1].estimated_tps:.0f} t/s")
    row("Est. TPS (batch=16)",
        lambda r: _batch16_tps(r[1].backend, r[1].estimated_tps))
    row("Est. VRAM",
        lambda r: f"{r[1].estimated_vram_gb:.1f} GB")
    row("Max concurrent reqs",
        lambda r: _max_concurrent(r[1].backend, r[1].params))
    row("Concurrency support",
        lambda r: _concurrency_note(r[1].backend))
    row("API compatibility",
        lambda r: _api_compat(r[1].backend))
    row("Quantization support",
        lambda r: _QUANT_SUPPORT.get(r[1].backend, "Unknown"))
    row("Multi-GPU",
        lambda r: _multi_gpu(r[1].backend))
    row("Setup complexity",
        lambda r: _SETUP_COMPLEXITY.get(r[1].backend, "Unknown"))
    row("Docker required",
        lambda r: _DOCKER_REQUIRED.get(r[1].backend, "Unknown"))

    console.print(table)

    has_sequential = any(
        b in ("ollama", "llamacpp") for b in results
    )
    if has_sequential:
        console.print(
            "[dim]  * Ollama and llama.cpp process one request at a time — "
            "TPS does not scale with concurrency.[/dim]"
        )


def _render_winner_section(results: dict) -> None:
    """
    Print the 'winner by use case' summary below the comparison table.

    Computes the best backend for each of four common use cases based on
    feasibility score, batching capability, VRAM usage, and backend name.

    Args:
        results: Dict mapping backend name → (FeasibilityScore, BackendConfig, strategy).
    """
    winners = _compute_use_case_winners(results)
    if not winners:
        return

    console.print("\n  [bold]Winner by use case:[/bold]")
    for use_case, backend in winners.items():
        console.print(f"  [green]✓[/green] {use_case:<30} [cyan]{backend}[/cyan]")


def _render_commands_section(results: dict) -> None:
    """
    Print the exact commands for each backend below the winners section.

    Args:
        results: Dict mapping backend name → (FeasibilityScore, BackendConfig, strategy).
    """
    console.print("\n  [bold]Commands:[/bold]")
    for backend, (score, cfg, _) in results.items():
        console.print(f"\n  [cyan]# {backend}[/cyan]")
        console.print(f"  {cfg.command}")


def _compute_use_case_winners(results: dict) -> dict:
    """
    Determine the best backend for each common use case.

    Use cases:
    - High-throughput API serving: highest TPS among backends that support batching.
    - Local interactive use: Ollama preferred, then llama.cpp.
    - CPU fallback / edge deploy: llama.cpp preferred, then Ollama.
    - Minimal VRAM usage: backend with the lowest estimated_vram_gb.

    Args:
        results: Dict mapping backend name → (FeasibilityScore, BackendConfig, strategy).

    Returns:
        Ordered dict of {use_case_label: winning_backend_name}.
    """
    winners = {}
    backend_names = list(results.keys())

    batching_backends = [b for b in backend_names if b in ("vllm", "tgi")]
    if batching_backends:
        best = max(batching_backends, key=lambda b: results[b][1].estimated_tps)
        winners["High-throughput API serving"] = best

    if "ollama" in backend_names:
        winners["Local interactive use"] = "ollama"
    elif "llamacpp" in backend_names:
        winners["Local interactive use"] = "llamacpp"

    if "llamacpp" in backend_names:
        winners["CPU fallback / edge deploy"] = "llamacpp"
    elif "ollama" in backend_names:
        winners["CPU fallback / edge deploy"] = "ollama"

    best_vram = min(backend_names, key=lambda b: results[b][1].estimated_vram_gb)
    winners["Minimal VRAM usage"] = best_vram

    return winners


def _batch16_tps(backend: str, base_tps: float) -> str:
    """
    Return the expected TPS at batch=16, with an asterisk for non-batching backends.

    vLLM and TGI use continuous batching so their throughput scales with concurrency.
    Ollama and llama.cpp are sequential — TPS is the same regardless of batch size.

    Args:
        backend: Backend name string.
        base_tps: Estimated TPS at batch=1 from the BackendConfig.

    Returns:
        Formatted string, e.g. "~310 t/s" or "~44 t/s *".
    """
    if backend in ("vllm", "tgi"):
        return f"~{base_tps * 3.5:.0f} t/s"
    return f"~{base_tps:.0f} t/s *"


def _max_concurrent(backend: str, params: dict) -> str:
    """
    Return a human-readable maximum concurrent requests estimate.

    For vLLM and TGI, max_num_seqs or a default of 32 is used.
    Sequential backends always report 1.

    Args:
        backend: Backend name string.
        params: BackendConfig.params dict.

    Returns:
        String like "~32" or "1 (no batching)".
    """
    if backend == "vllm":
        seqs = params.get("max_num_seqs", 32)
        return f"~{seqs}"
    if backend == "tgi":
        return "~32"
    return "1 (no batching)"


def _concurrency_note(backend: str) -> str:
    """Return a short concurrency capability label for the backend."""
    if backend == "vllm":
        return "✓ Full batching"
    if backend == "tgi":
        return "✓ Continuous batch"
    return "✗ Sequential"


def _api_compat(backend: str) -> str:
    """Return the API compatibility label for the backend."""
    if backend in ("vllm", "tgi", "ollama", "llamacpp"):
        return "OpenAI ✓"
    return "Custom"


def _multi_gpu(backend: str) -> str:
    """Return a multi-GPU capability label for the backend."""
    if backend == "vllm":
        return "✓ tensor parallel"
    if backend == "tgi":
        return "✓ sharding"
    return "✗"
