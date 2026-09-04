import json
import typer
from typing import Optional
from rich.console import Console

console = Console()
app = typer.Typer(invoke_without_command=True)


@app.callback(invoke_without_command=True)
def run(
    model_id: str = typer.Argument(...,
        help="HuggingFace model ID (e.g. 'meta-llama/Llama-3-8B'), "
             "GGUF ID (e.g. 'Qwen/Qwen2.5-7B-GGUF:Q4_K_M'), or local path"),

    gpuid: str = typer.Option("all", "--gpuid", "-g",
        help="GPU IDs to use: '0', '0,1,2', or 'all'"),

    backend: Optional[str] = typer.Option(None, "--backend", "-b",
        help="Force a specific backend: vllm | ollama | llamacpp | tgi"),

    output: str = typer.Option("table", "--output", "-o",
        help="Output format: table | json"),

    quant: bool = typer.Option(True, "--quant/--no-quant",
        help="Search for quantized alternatives if model doesn't fit"),

    verbose: bool = typer.Option(False, "--verbose", "-v",
        help="Show detailed estimation breakdown"),
):
    """
    Recommend optimal serving configuration for a model on your hardware.

    No model weights are downloaded — only config files (~5KB).

    Examples:
      mlfit recommend meta-llama/Llama-3-8B
      mlfit recommend Qwen/Qwen2.5-7B-GGUF:Q4_K_M
      mlfit recommend mistralai/Mistral-7B --backend vllm
      mlfit recommend meta-llama/Llama-3-8B --output json
    """
    from rich.progress import Progress, SpinnerColumn, TextColumn

    _backend = backend or "auto"

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        transient=True,
        console=console,
    ) as progress:
        hw_task = progress.add_task("Detecting hardware...", total=None)

        try:
            from mlfit.core.advisor import Advisor
            advisor = Advisor(model_id=model_id, gpuid=gpuid)

            # Trigger hardware detection
            _ = advisor.hardware
            progress.update(hw_task, description="[green]Detecting hardware... ✓[/green]")

            model_task = progress.add_task("Fetching model config...", total=None)
            # Trigger model analysis
            _ = advisor.model
            progress.update(model_task, description="[green]Fetching model config... ✓[/green]")

            result = advisor.recommend(backend=_backend)

        except Exception as e:
            progress.stop()
            _handle_error(e, model_id, _backend)
            raise typer.Exit(1)

    if output == "json":
        typer.echo(json.dumps(result.to_dict(), indent=2))
    else:
        from mlfit.reporter.table import render_recommend_result
        render_recommend_result(result)


def _handle_error(e: Exception, model_id: str, backend: str):
    """Print user-friendly error messages."""
    from mlfit.exceptions import (
        ModelNotFoundError, BackendNotInstalledError,
        InsufficientVRAMError, HardwareNotSupportedError,
    )

    if isinstance(e, ModelNotFoundError):
        console.print(f"\n[red]✗ Model not found:[/red] {model_id}")
        console.print(f"\n  Check the model exists at: https://huggingface.co/{model_id}")
        console.print("  Or verify the model ID spelling.")

    elif isinstance(e, BackendNotInstalledError):
        console.print(f"\n[red]✗ {e.backend} is not installed.[/red]")
        console.print(f"\n  Install with: [cyan]{e.install_command}[/cyan]")
        console.print(f"\n  Or use a different backend:")
        console.print(f"    [cyan]mlfit recommend {model_id}[/cyan]  (auto-select)")

    elif isinstance(e, InsufficientVRAMError):
        console.print(f"\n[red]✗ {model_id} requires {e.needed_gb:.1f} GB but only {e.available_gb:.1f} GB available.[/red]")
        console.print(f"\n  Run [cyan]mlfit recommend {model_id} --quant[/cyan] to see quantized alternatives.")

    elif isinstance(e, HardwareNotSupportedError):
        console.print(f"\n[red]✗ {e.backend} is not supported on this hardware.[/red]")
        console.print(f"\n  Run [cyan]mlfit recommend {model_id}[/cyan] to see compatible backends.")

    else:
        console.print(f"\n[red]✗ Error:[/red] {e}")
        console.print("\n[dim]Run with --verbose for more details.[/dim]")
