from textual.app import ComposeResult
from textual.message import Message
from textual.widget import Widget
from textual.widgets import Static, Button


class BackToBrowser(Message):
    """Posted when the user wants to return to the model browser."""


class ModelDetail(Widget):
    """
    Full-page detail view for a selected model.

    Shows architecture, size, fit result, recommendations, quantized alternatives,
    and the best run command. Press Esc or the Back button to return to the browser.
    """

    DEFAULT_CSS = """
    ModelDetail {
        height: 1fr;
        padding: 1;
        overflow-y: auto;
    }
    ModelDetail Button {
        margin-top: 1;
    }
    """

    BINDINGS = [("escape", "go_back", "Back"), ("b", "go_back", "Back")]

    can_focus = True

    def __init__(self, recommend_result=None, **kwargs) -> None:
        super().__init__(**kwargs)
        self._result = recommend_result

    def compose(self) -> ComposeResult:
        yield Static(self._build_text(), markup=True, id="detail-text")
        yield Button("← Back (Esc)", variant="default", id="back-btn")

    def action_go_back(self) -> None:
        self.post_message(BackToBrowser())

    def on_button_pressed(self, event: Button.Pressed) -> None:
        if event.button.id == "back-btn":
            self.post_message(BackToBrowser())

    def update_result(self, result) -> None:
        """Refresh the displayed content with a new RecommendResult."""
        self._result = result
        self.query_one("#detail-text", Static).update(self._build_text())

    def _build_text(self) -> str:
        if self._result is None:
            return "[$text-muted]Loading…[/]"

        r = self._result
        m = r.model
        lines = [
            f"[$primary bold]{m.model_id}[/]\n",
            f"[$accent bold]Architecture[/]   {m.architecture}",
            f"[$accent bold]Parameters[/]     {m.num_parameters / 1e9:.2f} B",
            f"[$accent bold]Precision[/]      {m.effective_dtype}  →  {m.estimated_size_gb:.1f} GB weights",
            f"[$accent bold]Context[/]        {m.max_context_length:,} tokens",
        ]

        if r.is_feasible:
            lines.append(
                f"[$accent bold]Fits on GPU[/]    [$success]✓ YES[/]  "
                f"({r.headroom_gb:.1f} GB headroom)"
            )
        else:
            lines.append(
                f"[$accent bold]Fits on GPU[/]    [$error]✗ NO[/]  "
                f"(need {r.estimated_vram_gb:.1f} GB)"
            )

        lines.append("\n[$accent bold]Recommendations[/]")
        if r.configs:
            for i, (score, cfg, _) in enumerate(r.configs[:4], 1):
                bar = "█" * int(score.score * 10) + "░" * (10 - int(score.score * 10))
                lines.append(
                    f"  {i}. [$primary]{cfg.backend:<10}[/] {bar} "
                    f"{score.score * 100:.0f}%   ~{cfg.estimated_tps:.0f} t/s"
                )
        else:
            lines.append("  [$text-muted]No compatible backends found.[/]")

        if r.quantized_alternatives:
            lines.append("\n[$accent bold]Quantized Alternatives[/]")
            for alt in r.quantized_alternatives[:4]:
                lines.append(
                    f"  • {alt.model_id}  {alt.size_gb:.1f} GB  "
                    f"[$text-muted]{alt.quality_loss} loss[/]"
                )

        if r.best:
            lines.append(f"\n[$accent bold]Best Command[/]\n[$success]{r.best.command}[/]")

        return "\n".join(lines)
