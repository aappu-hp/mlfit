import platform

from textual.app import ComposeResult
from textual.widget import Widget
from textual.widgets import Static

from mlfit._version import __version__

_SEP = "  [$text-muted]│[/]  "


class HardwareBar(Widget):
    """
    Four-row header at the top of the TUI:

      Row 1 — compute:  GPU/unified memory (+ API) · CPU cores/threads · AVX.
      Row 2 — memory:   available/total RAM · swap · free disk.
      Row 3 — runtimes: Ollama / llama.cpp / vLLM / MLX / LM Studio.
      Row 4 — status:   mlfit version · OS · cached models · LLM provider · theme.

    Each row is updated independently as its data becomes available; the widget
    re-renders all four lines on any change.
    """

    DEFAULT_CSS = """
    HardwareBar {
        height: 4;
        padding: 0 1;
    }
    """

    def __init__(self, hardware_profile=None, **kwargs) -> None:
        super().__init__(**kwargs)
        self._hw = hardware_profile
        self._runtimes: dict[str, bool] = {}
        self._model_count: int = 0
        self._provider_name: str = "…"
        self._theme_name: str = "…"

    def compose(self) -> ComposeResult:
        yield Static(self._build_text(), markup=True, id="hw-bar-text")

    def update_hardware(self, hardware_profile) -> None:
        """Set the detected hardware profile (rows 1–2) and re-render."""
        self._hw = hardware_profile
        self._refresh()

    def update_runtimes(self, runtimes: dict[str, bool]) -> None:
        """Set the detected runtime availability map (row 3) and re-render."""
        self._runtimes = runtimes
        self._refresh()

    def update_status(self, model_count: int, provider_name: str, theme_name: str) -> None:
        """Set the status line values (row 4) and re-render."""
        self._model_count = model_count
        self._provider_name = provider_name
        self._theme_name = theme_name
        self._refresh()

    def _refresh(self) -> None:
        self.query_one("#hw-bar-text", Static).update(self._build_text())

    def _build_text(self) -> str:
        return "\n".join(
            (
                self._compute_line(),
                self._memory_line(),
                self._runtime_line(),
                self._status_line(),
            )
        )

    def _compute_line(self) -> str:
        if self._hw is None:
            return "[$text-muted]Detecting hardware…[/]"

        hw = self._hw
        parts = []

        if hw.gpus:
            gpu = hw.gpus[0]
            api = self._gpu_api(hw.gpu_backend, gpu.compute_capability)
            extra = f" +{len(hw.gpus) - 1}" if len(hw.gpus) > 1 else ""
            if gpu.is_unified_memory:
                label = "Unified"
                mem = f"{gpu.vram_total_gb:.0f}GB shared"
            else:
                label = "GPU"
                mem = f"{gpu.vram_free_gb:.1f}/{gpu.vram_total_gb:.0f}GB"
            parts.append(
                f"[$accent bold]{label}:[/] {gpu.name} · {mem}{extra} · [$primary]{api}[/]"
            )
        else:
            parts.append("[$text-muted]GPU: None · CPU[/]")

        parts.append(
            f"[$accent bold]CPU:[/] {hw.cpu_cores}c/{hw.cpu_threads}t · {self._avx_label(hw)}"
        )
        return _SEP.join(parts)

    def _memory_line(self) -> str:
        if self._hw is None:
            return ""

        hw = self._hw
        parts = [
            f"[$accent bold]RAM:[/] {hw.ram_available_gb:.1f}/{hw.ram_gb:.0f}GB",
        ]
        if hw.swap_total_gb > 0:
            parts.append(f"[$accent bold]Swap:[/] {hw.swap_used_gb:.1f}/{hw.swap_total_gb:.0f}GB")
        parts.append(f"[$accent bold]Disk:[/] {hw.disk_free_gb:.0f}GB free")
        return _SEP.join(parts)

    def _runtime_line(self) -> str:
        if not self._runtimes:
            return "[$text-muted]Detecting runtimes…[/]"

        parts = []
        for name, available in self._runtimes.items():
            mark = "[$success]✓[/]" if available else "[$error]✗[/]"
            parts.append(f"{name} {mark}")
        return _SEP.join(parts)

    def _status_line(self) -> str:
        provider = self._provider_name
        provider_marked = (
            f"[$success]{provider} ✓[/]" if provider not in ("None", "No LLM", "…")
            else f"[$text-muted]{provider}[/]"
        )
        parts = [
            f"[$primary bold]mlfit v{__version__}[/]",
            f"[$text-muted]{self._os_label()}[/]",
            f"[$accent bold]{self._model_count}[/] models",
            f"[$accent bold]LLM:[/] {provider_marked}",
            f"[$accent bold]Theme:[/] {self._theme_name}",
        ]
        return _SEP.join(parts)

    @staticmethod
    def _gpu_api(backend: str, compute_capability) -> str:
        if backend == "mps":
            return "Metal"
        if backend == "cuda":
            if compute_capability and compute_capability != (0, 0):
                return f"CUDA {compute_capability[0]}.{compute_capability[1]}"
            return "CUDA"
        if backend == "rocm":
            return "ROCm"
        return "CPU"

    @staticmethod
    def _os_label() -> str:
        system = platform.system()
        if system == "Darwin":
            version = platform.mac_ver()[0]
            base = f"macOS {version}" if version else "macOS"
        else:
            base = f"{system} {platform.release()}".strip()
        return f"{base} · {platform.machine()}"

    @staticmethod
    def _avx_label(hw) -> str:
        if hw.has_avx512:
            return "AVX512"
        if hw.has_avx2:
            return "AVX2"
        if hw.has_avx:
            return "AVX"
        return "no-AVX"
