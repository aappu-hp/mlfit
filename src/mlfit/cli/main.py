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
    no_args_is_help=True,
)

# Register subcommands
from mlfit.cli.commands import recommend, profile, compare, serve  # noqa: E402

app.add_typer(recommend.app, name="recommend",
              help="Recommend optimal serving config (no weight download)")
app.command("profile")(profile.run)
app.command("serve")(serve.run)
app.command("compare")(compare.run)


if __name__ == "__main__":
    app()
