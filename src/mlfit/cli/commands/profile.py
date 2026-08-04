import json
import time
import typer
from rich.console import Console
from rich.live import Live
from rich.spinner import Spinner
from rich.text import Text

console = Console()

_PHASE_LABELS = {
    "searching": "Finding optimal config",
    "warmup":    "Warming up",
    "benchmarking": "Benchmarking",
}


def run(
    model_id: str = typer.Argument(...),
    backend: str = typer.Option("auto", "--backend", "-b",
        help="Backend to profile: vllm | ollama | llamacpp | tgi | auto"),
    gpuid: str = typer.Option("all", "--gpuid", "-g",
        help="GPU IDs to use: '0', '0,1,2', or 'all'"),
    concurrency: str = typer.Option("1,4,16", "--concurrency", "-c",
        help="Comma-separated concurrency levels to benchmark"),
    output: str = typer.Option("table", "--output", "-o",
        help="Output format: table | json"),
):
    """
    Profile actual memory usage and benchmark throughput.

    Downloads and loads the model — may take several minutes.
    The server is started automatically and stopped when profiling finishes.

    Examples:
      mlfit profile mistralai/Mistral-7B
      mlfit profile Qwen/Qwen2.5-7B --backend ollama --concurrency 1,4
      mlfit profile mistralai/Mistral-7B --output json
    """
    concurrency_levels = [int(c.strip()) for c in concurrency.split(",")]

    console.print(
        f"\n[yellow]⚠[/yellow]  This command loads the model — "
        "weights will be downloaded if not cached.\n"
        "     Press [bold]Ctrl+C[/bold] to cancel at any time.\n"
    )

    from mlfit.core.advisor import Advisor
    advisor = Advisor(model_id=model_id, gpuid=gpuid)

    phase_state = {"phase": "", "message": "", "is_success": True}

    def on_progress(phase: str, message: str, is_success: bool) -> None:
        phase_state.update(phase=phase, message=message, is_success=is_success)
        icon = "[green]✓[/green]" if is_success else "[red]✗[/red]"
        label = _PHASE_LABELS.get(phase, phase.capitalize())
        console.print(f"  {icon}  [dim]{label}:[/dim] {message}")

    try:
        result = advisor.profile(
            backend=backend,
            concurrency=concurrency_levels,
            progress_callback=on_progress,
        )
    except KeyboardInterrupt:
        console.print("\n[yellow]Profiling cancelled.[/yellow]")
        raise typer.Exit(0)
    except Exception as exc:
        console.print(f"\n[red]✗ Profiling failed:[/red] {exc}")
        raise typer.Exit(1)

    if output == "json":
        typer.echo(json.dumps(result.to_dict(), indent=2))
    else:
        from mlfit.reporter.table import render_profile_result
        render_profile_result(result)
