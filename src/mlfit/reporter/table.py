from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich import box

from mlfit._version import __version__

console = Console()


def render_header() -> None:
    """Print the mlfit banner to the console."""
    console.print(Panel(
        f"[bold cyan]mlfit v{__version__}[/bold cyan] — Universal Model Deployment Advisor",
        border_style="cyan",
        padding=(0, 2),
    ))


def render_hardware_panel(hardware) -> None:
    """Render the hardware summary panel to the console.

    Args:
        hardware: HardwareProfile containing GPU, CPU, RAM, and disk information.
    """
    if hardware.gpus:
        if hardware.gpu_backend == "mps":
            g = hardware.gpus[0]
            gpu_line = (
                f"[bold]GPU[/bold]   {g.name} · Unified Memory · "
                f"{g.vram_total_gb:.1f} GB total · {g.vram_free_gb:.1f} GB free\n"
                f"       [dim](unified = RAM and GPU memory are the same pool)[/dim]"
            )
        else:
            gpu_lines = []
            for g in hardware.gpus:
                gpu_lines.append(
                    f"[bold]GPU[/bold]   {g.name} · "
                    f"{g.vram_total_gb:.1f} GB total · {g.vram_free_gb:.1f} GB free"
                )
            gpu_line = "\n".join(gpu_lines)
    else:
        gpu_line = "[bold]GPU[/bold]   [dim]None detected[/dim]"

    if hardware.has_avx512:
        avx_flag = "AVX512"
    elif hardware.has_avx2:
        avx_flag = "AVX2"
    elif hardware.has_avx:
        avx_flag = "AVX"
    else:
        avx_flag = "none"

    content = (
        f"{gpu_line}\n"
        f"[bold]CPU[/bold]   {hardware.cpu_cores} cores / {hardware.cpu_threads} threads · {avx_flag}\n"
        f"[bold]RAM[/bold]   {hardware.ram_gb:.1f} GB system memory\n"
        f"[bold]Disk[/bold]  {hardware.disk_free_gb:.0f} GB free"
    )
    console.print(Panel(content, title="[bold]Hardware[/bold]", border_style="blue"))


def render_model_panel(model, fits: bool, estimated_vram: float, headroom: float) -> None:
    """Render the model analysis panel to the console.

    Args:
        model: ModelProfile with architecture and size metadata.
        fits: Whether the model fits in available VRAM or RAM.
        estimated_vram: Predicted VRAM requirement in GB.
        headroom: Available VRAM/RAM minus estimated requirement, in GB.
    """
    fits_color = "green" if fits else "red"
    fits_icon = "✓ YES" if fits else "✗ NO"

    if fits:
        fits_detail = f"({estimated_vram:.1f} GB needed, {headroom:.1f} GB headroom)"
    else:
        fits_detail = f"(need ~{estimated_vram:.1f} GB, only {estimated_vram - headroom:.1f} GB available)"

    if model.model_type == "gguf":
        content = (
            f"[bold]Type[/bold]          GGUF · {model.quant_type or 'Q4_K_M'}\n"
            f"[bold]Parameters[/bold]    {model.num_parameters / 1e9:.2f} B\n"
            f"[bold]File size[/bold]     {model.estimated_size_gb:.2f} GB\n"
            f"[bold]Context[/bold]       {model.max_context_length:,} tokens\n"
            f"[bold]Fits[/bold]          [{fits_color}]{fits_icon}[/{fits_color}]  {fits_detail}"
        )
    else:
        content = (
            f"[bold]Architecture[/bold]  {model.architecture}\n"
            f"[bold]Parameters[/bold]    {model.num_parameters / 1e9:.2f} B\n"
            f"[bold]Precision[/bold]     {model.dtype}  →  {model.estimated_size_gb:.1f} GB weights\n"
            f"[bold]Context[/bold]       {model.max_context_length:,} tokens (max)\n"
            f"[bold]Est. VRAM[/bold]     {estimated_vram:.1f} GB  (weights + KV cache + overhead)\n"
            f"[bold]Fits on GPU[/bold]   [{fits_color}]{fits_icon}[/{fits_color}]  {fits_detail}"
        )

    console.print(Panel(
        content,
        title=f"[bold]Model: {model.model_id}[/bold]",
        border_style="green",
    ))


def render_configs_table(configs, hardware) -> None:
    """Render the ranked backend configurations table to the console.

    Args:
        configs: List of (FeasibilityScore, BackendConfig, BaseStrategy) tuples,
                 sorted descending by feasibility score.
        hardware: HardwareProfile — used to show hardware-specific warnings.
    """
    if not configs:
        console.print("[yellow]⚠  No compatible backends found for this hardware.[/yellow]")
        return

    table = Table(
        title="Recommended Configurations",
        box=box.ROUNDED,
        show_header=True,
        header_style="bold",
    )
    table.add_column("#", style="dim", width=3)
    table.add_column("Backend", style="bold cyan", width=12)
    table.add_column("Feasibility", width=20)
    table.add_column("Est. TPS", justify="right", style="yellow", width=10)
    table.add_column("Est. VRAM", justify="right", width=10)
    table.add_column("Key Settings", style="dim")

    for i, (score, cfg, strategy) in enumerate(configs, 1):
        filled = int(score.score * 9)
        bar = "█" * filled + "░" * (9 - filled)
        feasibility = f"{bar} {score.score * 100:.0f}%"
        key_settings = strategy.format_key_settings(cfg.params)
        table.add_row(
            str(i),
            cfg.backend,
            feasibility,
            f"~{cfg.estimated_tps:.0f} t/s",
            f"{cfg.estimated_vram_gb:.1f} GB",
            key_settings,
        )

    console.print(table)

    if hardware.gpu_backend == "mps":
        console.print(
            "\n[yellow]⚠  vLLM and TGI are not available on Apple Silicon (CUDA only).[/yellow]"
        )


def render_best_command(config) -> None:
    """Render the 'Best Command' panel showing the top-ranked deploy command.

    Args:
        config: BackendConfig whose command field is displayed.
    """
    console.print(Panel(
        f"[bold green]{config.command}[/bold green]",
        title=f"[bold]✓ Best Command ({config.backend})[/bold]",
        border_style="green",
    ))


def render_quant_alternatives(alternatives) -> None:
    """
    Render quantized and offload alternatives when the model does not fit.

    Splits the alternatives into two sections:
      - Quantized/offload variants (GGUF, AWQ, GPTQ, CPU offload)
      - A plain-text hint for each entry with a ready-to-run command

    Args:
        alternatives: List of AlternativeModel instances from the quantization advisor.
    """
    if not alternatives:
        return

    offload = [a for a in alternatives if a.quant_type is None]
    quantized = [a for a in alternatives if a.quant_type is not None]

    if quantized:
        console.print(
            "\n[bold yellow]⚠  Model doesn't fit. "
            "Quantized alternatives that may fit:[/bold yellow]"
        )
        table = Table(box=box.SIMPLE, show_header=True)
        table.add_column("#", style="dim", width=3)
        table.add_column("Model ID", style="cyan")
        table.add_column("Size", justify="right", width=8)
        table.add_column("Quality loss", style="green")
        table.add_column("Backend")

        for i, alt in enumerate(quantized, 1):
            label = alt.quality_loss.split("—")[0].strip() if "—" in alt.quality_loss else alt.quality_loss
            table.add_row(
                str(i),
                alt.model_id,
                f"{alt.size_gb:.1f} GB",
                label,
                alt.backend,
            )
        console.print(table)

    if offload:
        console.print("\n[bold]CPU + GPU offload option:[/bold]")
        for alt in offload:
            console.print(f"  [cyan]•[/cyan] {alt.backend}")
            if alt.notes:
                console.print(f"    [dim]{alt.notes}[/dim]")

    if quantized:
        best = quantized[0]
        console.print(
            f"\n[dim]▶  Best fit: [cyan]mlfit recommend {best.model_id}[/cyan][/dim]"
        )


def render_profile_result(result) -> None:
    """
    Render a ProfilingResult to the terminal.

    Displays a memory panel, a latency panel, and a throughput table
    followed by the verified optimal command.

    Args:
        result: ProfilingResult from a completed ProfileSession.
    """
    render_header()

    hardware = result.hardware
    if hardware.gpus:
        gpu_line = f"[bold]GPU[/bold]   {hardware.gpus[0].name}"
        if hardware.gpu_backend == "mps":
            gpu_line += " · Unified Memory"
        else:
            gpu_line += f" · {hardware.gpus[0].vram_total_gb:.0f} GB"
    else:
        gpu_line = "[bold]GPU[/bold]   [dim]None[/dim]"

    hw_content = (
        f"{gpu_line}\n"
        f"[bold]RAM[/bold]   {hardware.ram_gb:.1f} GB"
    )
    console.print(Panel(
        hw_content,
        title=f"[bold]Profile: {result.model_id}  ·  {result.backend}[/bold]",
        border_style="blue",
    ))

    vram_efficiency = (
        result.peak_vram_gb / hardware.gpus[0].vram_total_gb * 100
        if hardware.gpus else 0.0
    )
    mem_content = (
        f"[bold]Peak VRAM used[/bold]         {result.peak_vram_gb:.2f} GB\n"
        f"[bold]VRAM efficiency[/bold]        {vram_efficiency:.1f}%\n"
        f"[bold]gpu_memory_utilization[/bold] "
        f"{result.final_config.params.get('gpu_memory_utilization', 'n/a')}\n"
        f"[bold]max_model_len[/bold]          "
        f"{result.final_config.params.get('max_model_len', 'n/a')}"
    )
    console.print(Panel(mem_content, title="[bold]Memory[/bold]", border_style="cyan"))

    if result.benchmark_points:
        first = result.benchmark_points[0]
        lat_content = (
            f"[bold]TTFT p50 (median)[/bold]   {first.ttft_p50_ms:.0f} ms\n"
            f"[bold]TTFT p95[/bold]            {first.ttft_p95_ms:.0f} ms"
        )
        console.print(Panel(
            lat_content, title="[bold]Latency (Time to First Token)[/bold]",
            border_style="yellow",
        ))

    tput_table = Table(
        title="Throughput (Tokens / Second)",
        box=box.SIMPLE,
        show_header=True,
    )
    tput_table.add_column("Concurrency", style="cyan")
    tput_table.add_column("Tokens / sec", justify="right", style="bold green")

    for bp in result.benchmark_points:
        tput_table.add_row(str(bp.concurrency), f"{bp.tps:.1f} t/s")

    console.print(tput_table)

    console.print(Panel(
        f"[bold green]{result.final_config.command}[/bold green]",
        title="[bold]✓ Verified Optimal Command[/bold]",
        border_style="green",
    ))

    console.print(
        f"\n[dim]Elapsed: {result.time_elapsed_s:.0f}s  ·  "
        f"Timestamp: {result.timestamp}[/dim]"
    )


def render_recommend_result(result) -> None:
    """Render the full recommendation output: header, hardware, model, configs, and command.

    Args:
        result: RecommendResult from Advisor.recommend().
    """
    render_header()

    hardware = result.hardware
    model = result.model

    render_hardware_panel(hardware)
    render_model_panel(model, result.is_feasible, result.estimated_vram_gb, result.headroom_gb)

    render_configs_table(result.configs, hardware)

    if result.best:
        render_best_command(result.best)

    if not result.is_feasible and result.quantized_alternatives:
        render_quant_alternatives(result.quantized_alternatives)

    console.print(f"\n[dim]ℹ  Run [cyan]`mlfit profile {model.model_id}`[/cyan] for actual benchmark numbers.[/dim]")
    if result.configs and len(result.configs) > 1:
        backends = ",".join(c[1].backend for c in result.configs[:3])
        console.print(
            f"[dim]ℹ  Run [cyan]`mlfit compare {model.model_id} --backends {backends}`[/cyan] to compare backends.[/dim]"
        )
