import typer
from typing import Optional
from rich.console import Console

console = Console()


def run(
    model_id: str = typer.Argument(...),
    backend: str = typer.Option("auto", "--backend", "-b"),
    gpuid: str = typer.Option("all", "--gpuid", "-g"),
    port: int = typer.Option(8000, "--port", "-p"),
):
    """
    Recommend and show the command to serve this model optimally.

    Full auto-serve with profiling is a Phase 2 feature.
    Currently shows the recommended serve command.
    """
    from mlfit.core.advisor import Advisor
    from mlfit.reporter.table import render_header, render_best_command

    advisor = Advisor(model_id=model_id, gpuid=gpuid)
    result = advisor.recommend(backend=backend)

    render_header()

    if not result.configs:
        console.print("[red]✗ No compatible backends found for this hardware.[/red]")
        raise typer.Exit(1)

    best = result.best
    console.print(f"\n  [bold]Optimal serving command for[/bold] [cyan]{model_id}[/cyan]:\n")
    render_best_command(best)

    console.print(f"\n[dim]ℹ  This is a static recommendation. Run [cyan]mlfit profile {model_id}[/cyan] to benchmark before serving.[/dim]")
