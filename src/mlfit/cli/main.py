import typer

app = typer.Typer(
    name="mlfit",
    help=(
        "[bold cyan]mlfit[/bold cyan] — Universal model deployment advisor.\n\n"
        "Find the best way to serve any AI model on your hardware.\n"
        "No GPU expertise required."
    ),
    add_completion=False,
    rich_markup_mode="rich",
    invoke_without_command=True,
)

# Register subcommands
from mlfit.cli.commands import recommend, profile, compare, serve  # noqa: E402

app.add_typer(recommend.app, name="recommend",
              help="Recommend optimal serving config (no weight download)")
app.command("profile")(profile.run)
app.command("serve")(serve.run)
app.command("compare")(compare.run)


@app.callback()
def _default(ctx: typer.Context, gpuid: str = typer.Option("all", hidden=True)) -> None:
    """Open the interactive TUI when no subcommand is given."""
    from mlfit.cache.bootstrap import bootstrap_if_needed
    bootstrap_if_needed()

    if ctx.invoked_subcommand is None:
        from mlfit.tui.app import MLFitApp
        # mouse=True keeps mouse capture on so the scroll wheel scrolls the TUI
        # (not the host terminal). Clicks are made inert inside MLFitApp so the
        # app stays keyboard-driven, matching llmfit.
        MLFitApp(gpuid=gpuid).run(mouse=True)


if __name__ == "__main__":
    app()
