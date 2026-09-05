import logging
import pathlib

from mlfit.cache.model_cache import SEED_MODELS, ModelCache

logger = logging.getLogger(__name__)


def bootstrap_if_needed(cache: ModelCache | None = None) -> bool:
    """
    Initialize ~/.mlfit/ on first run and pre-fetch seed model configs.

    Silently skips if the cache directory already exists. Returns True when
    bootstrap was performed, False when skipped (already initialised).

    Args:
        cache: ModelCache instance to use. Defaults to the shared instance.

    Returns:
        True if bootstrap was performed, False if already initialised.
    """
    from mlfit.cache.model_cache import get_default_cache

    cache = cache or get_default_cache()
    mlfit_dir = pathlib.Path.home() / ".mlfit"

    if mlfit_dir.exists():
        return False

    _run_bootstrap(cache)
    return True


def _run_bootstrap(cache: ModelCache) -> None:
    from rich.console import Console
    from rich.progress import Progress, SpinnerColumn, TextColumn

    console = Console(stderr=True)
    console.print(
        "[bold cyan]mlfit[/bold cyan] Initializing — fetching popular model configs...",
    )

    succeeded = 0
    failed = []

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        console=console,
        transient=True,
    ) as progress:
        task = progress.add_task("Fetching seed models...", total=len(SEED_MODELS))
        for model_id in SEED_MODELS:
            progress.update(task, description=f"Fetching {model_id}...")
            try:
                _fetch_and_cache(model_id, cache)
                succeeded += 1
            except Exception as exc:
                logger.debug("Seed fetch failed for %s: %s", model_id, exc)
                failed.append(model_id)
            progress.advance(task)

    console.print(
        f"[green]✓[/green] Cached {succeeded}/{len(SEED_MODELS)} model configs "
        f"to [dim]{cache.cache_dir}[/dim]"
    )
    if failed:
        console.print(
            f"[yellow]  {len(failed)} skipped (gated/private): "
            + ", ".join(failed[:3])
            + ("..." if len(failed) > 3 else "")
            + "[/yellow]"
        )


def _fetch_and_cache(model_id: str, cache: ModelCache) -> None:
    """Fetch config.json (+ exact param count) from HF Hub and cache it."""
    import json
    from huggingface_hub import hf_hub_download

    from mlfit.analyzers.hf_analyzer import fetch_param_count

    config_path = hf_hub_download(repo_id=model_id, filename="config.json")
    with open(config_path) as f:
        config = json.load(f)

    exact = fetch_param_count(model_id)
    if exact:
        config["_mlfit_params_total"] = exact

    cache.save(model_id, config)
