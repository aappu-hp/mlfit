from textual import events
from textual.app import App, ComposeResult
from textual.containers import Horizontal
from textual.reactive import reactive

from mlfit.tui.widgets.hardware_panel import HardwareBar
from mlfit.tui.widgets.left_panel import LeftPanel
from mlfit.tui.widgets.chat_panel import ChatPanel
from mlfit.tui.widgets.status_bar import StatusBar
from mlfit.tui.theme import ThemeManager
from mlfit._version import __version__


class MLFitApp(App):
    """
    Full-screen terminal UI for mlfit.

    Layout (top to bottom):
      HardwareBar  — compact 1-line hardware summary, always visible
      Horizontal   — left panel (model browser/detail) + right panel (chat), each 1fr
      StatusBar    — keyboard shortcut hints, always visible

    Colors are driven entirely by the active Textual theme (semantic tokens),
    so pressing `t` re-skins the whole UI and the choice is persisted to disk.
    Both the left and chat panels are independently toggleable via keyboard.
    """

    TITLE = f"mlfit v{__version__} — Universal Model Deployment Advisor"
    ENABLE_COMMAND_PALETTE = True

    CSS = """
    /* NB: don't set `background` on the bare `Screen` selector — it also matches
       HelpScreen (a ModalScreen) and would override its translucent overlay,
       making the help look like a full new screen instead of a popup. */
    Screen {
        layout: vertical;
    }

    /* Header: 4 rows (compute / memory / runtimes / status), with a single rule. */
    HardwareBar {
        height: 4;
        color: $foreground;
        padding: 0 1;
        border-bottom: solid $surface-lighten-1;
    }

    #main-area {
        height: 1fr;
        layout: horizontal;
    }

    /* Flat panels: no box borders — just a single vertical divider between
       them. The divider brightens on the left panel when its content is focused. */
    LeftPanel, ChatPanel {
        width: 1fr;
        background: $background;
        padding: 0 1;
    }
    LeftPanel {
        border-right: solid $surface-lighten-1;
    }
    LeftPanel:focus-within {
        border-right: solid $primary;
    }

    /* Inputs: subtle until focused. */
    Input {
        border: round $surface-lighten-1;
        background: $surface;
    }
    Input:focus {
        border: round $accent;
    }

    /* Lists / logs: no box, highlighted row uses the theme's block cursor. */
    ListView, RichLog, VerticalScroll {
        background: transparent;
        scrollbar-size-vertical: 1;
    }
    """

    BINDINGS = [
        ("i", "focus_chat", "Message"),
        ("tab", "switch_focus", "Focus"),
        ("l", "toggle_left", "Left"),
        ("c", "toggle_chat", "Chat"),
        ("t", "cycle_theme", "Theme"),
        ("h", "help", "Help"),
        ("q", "quit", "Quit"),
        ("ctrl+c", "quit", "Quit"),
    ]

    show_left: reactive[bool] = reactive(True)
    show_chat: reactive[bool] = reactive(False)

    def __init__(self, gpuid: str = "all", **kwargs) -> None:
        super().__init__(**kwargs)
        self.gpuid = gpuid
        self._hw = None
        self._model_ids: list[str] = []
        self._provider_name: str | None = None
        self._theme_manager = ThemeManager()

    def compose(self) -> ComposeResult:
        yield HardwareBar(id="hw-bar")
        with Horizontal(id="main-area"):
            yield LeftPanel(hardware_profile=None, id="left-panel")
            yield ChatPanel(hardware_profile=None, id="chat-panel")
        yield StatusBar(id="status-bar")

    def on_mount(self) -> None:
        self.theme = self._theme_manager.load()
        self.run_worker(self._detect_hardware_and_models(), exclusive=True)

    async def on_event(self, event: events.Event) -> None:
        """
        Keep mouse capture on (so the scroll wheel scrolls the TUI rather than
        the host terminal) while making pointer clicks inert — the app is
        keyboard-driven, matching llmfit.

        Mouse button presses/releases are swallowed here; scroll and movement
        events fall through to the default handler so wheel scrolling still works.
        """
        if isinstance(event, (events.MouseDown, events.MouseUp)):
            return
        await super().on_event(event)

    def action_cycle_theme(self) -> None:
        """Switch to the next theme, remember it, and refresh the header label."""
        next_theme = self._theme_manager.next_theme(self.theme)
        self.theme = next_theme
        self._theme_manager.save(next_theme)
        self._refresh_status_line()

    def action_help(self) -> None:
        """Toggle the key-bindings overlay."""
        from mlfit.tui.screens.help_screen import HelpScreen

        if isinstance(self.screen, HelpScreen):
            self.pop_screen()
        else:
            self.push_screen(HelpScreen())

    def action_focus_chat(self) -> None:
        """Focus the chat input (vim insert), revealing the panel if hidden."""
        if not self.show_chat:
            self.show_chat = True
        self.query_one("#chat-input").focus()

    def action_switch_focus(self) -> None:
        """Toggle focus between the model list and the chat input."""
        chat_input = self.query_one("#chat-input")
        model_list = self.query_one("#model-list")
        if self.focused is chat_input and self.show_left:
            model_list.focus()
        elif self.show_chat:
            chat_input.focus()
        elif self.show_left:
            model_list.focus()

    def watch_show_left(self, value: bool) -> None:
        self.query_one("#left-panel", LeftPanel).display = value

    def watch_show_chat(self, value: bool) -> None:
        self.query_one("#chat-panel", ChatPanel).display = value

    def action_toggle_left(self) -> None:
        # Never leave the screen empty: if hiding left would hide the last
        # visible panel, reveal chat instead of blanking.
        if self.show_left and not self.show_chat:
            self.show_chat = True
        self.show_left = not self.show_left
        self._normalize_focus()

    def action_toggle_chat(self) -> None:
        if self.show_chat and not self.show_left:
            self.show_left = True
        self.show_chat = not self.show_chat
        self._normalize_focus()

    def _normalize_focus(self) -> None:
        """
        Keep focus on a non-input widget after a panel toggle so single-key
        shortcuts keep working. Prefers the model list; otherwise clears focus
        (app-level bindings still fire when nothing is focused).
        """
        if self.show_left:
            try:
                self.query_one("#model-list").focus()
                return
            except Exception:
                pass
        self.set_focus(None)

    async def _detect_hardware_and_models(self) -> None:
        import asyncio
        from mlfit.detectors import detect_hardware
        from mlfit.detectors.runtimes import detect_runtimes
        from mlfit.cache.model_cache import get_default_cache

        try:
            self._hw = await asyncio.to_thread(detect_hardware, self.gpuid)
        except Exception:
            self._hw = None

        hw_bar = self.query_one("#hw-bar", HardwareBar)
        hw_bar.update_hardware(self._hw)
        hw_bar.update_runtimes(await asyncio.to_thread(detect_runtimes))

        left = self.query_one("#left-panel", LeftPanel)
        left._hw = self._hw
        self._load_rows()

        chat = self.query_one("#chat-panel", ChatPanel)
        chat._hw = self._hw
        chat._system_prompt = chat._build_system_prompt()

        self._refresh_status_line()
        self.run_worker(self._enrich_params(), exclusive=False)

    def _load_rows(self) -> None:
        """Read the cache and build display rows for the model table."""
        from mlfit.cache.model_cache import get_default_cache
        from mlfit.tui.model_meta import build_row

        cache = get_default_cache()
        self._model_ids = cache.list_cached()
        capacity = self._fit_capacity_gb()
        rows = [build_row(mid, cache.get(mid), capacity) for mid in self._model_ids]
        self.query_one("#left-panel", LeftPanel).set_models(rows)

    def _fit_capacity_gb(self) -> float:
        """
        Total memory capacity to judge model fit against.

        Uses total GPU/unified-memory capacity (not transient free memory) so a
        model's fit reflects what the machine can hold, not the moment's free RAM.
        Falls back to system RAM when there is no GPU.
        """
        if not self._hw:
            return 0.0
        if self._hw.gpus:
            return sum(g.vram_total_gb for g in self._hw.gpus)
        return self._hw.ram_gb

    def refresh_models_from_cache(self) -> None:
        """Rebuild the model table from the cache and update the header count."""
        self._load_rows()
        self._refresh_status_line()

    async def _enrich_params(self) -> None:
        """
        Fill exact parameter counts for already-cached models that predate the
        safetensors enrichment. Best-effort and off the UI thread; silently
        no-ops when offline.
        """
        import asyncio
        from mlfit.analyzers.hf_analyzer import fetch_param_count
        from mlfit.cache.model_cache import get_default_cache

        cache = get_default_cache()
        updated = False
        for model_id in list(self._model_ids):
            config = cache.get(model_id)
            if not config or config.get("_mlfit_params_total"):
                continue
            total = await asyncio.to_thread(fetch_param_count, model_id)
            if total:
                config["_mlfit_params_total"] = total
                cache.save(model_id, config)
                updated = True

        if updated:
            self.refresh_models_from_cache()

    def _refresh_status_line(self) -> None:
        """Update the header status row (cached models, LLM provider, theme)."""
        if self._provider_name is None:
            from mlfit.tui.chat.base import resolve_provider

            provider = resolve_provider()
            self._provider_name = (
                type(provider).__name__.replace("Provider", "") if provider else "None"
            )
        self.query_one("#hw-bar", HardwareBar).update_status(
            model_count=len(self._model_ids),
            provider_name=self._provider_name,
            theme_name=self.theme,
        )

    def post_message_to_chat(self, model_id: str, recommend_result=None) -> None:
        """Called by LeftPanel when a model is selected so chat panel gets context."""
        self.query_one("#chat-panel", ChatPanel).update_context(model_id, recommend_result)
