from rich.markup import escape

from textual.app import ComposeResult
from textual.containers import VerticalScroll
from textual.widget import Widget
from textual.widgets import Input, Static


_NO_KEY_MSG = (
    "LLM chat is disabled.\n\n"
    "Set [bold]GEMINI_API_KEY[/bold] in your .env to enable Gemini chat.\n"
    "Or set [bold]AZURE_OPENAI_API_KEY[/bold] + [bold]AZURE_OPENAI_ENDPOINT[/bold] for Azure.\n\n"
    "Type [bold]/history[/bold] to browse past sessions."
)

_USER_PREFIX = "[$primary bold]You[/]"
_ASSISTANT_PREFIX = "[$success bold]mlfit[/]"


class ChatMessage(Static):
    """A single chat message row, styled by role via a CSS class."""

    def __init__(self, role: str, prefix: str, body: str) -> None:
        super().__init__(f"{prefix}  {escape(body)}")
        self.add_class(f"msg-{role}")


class ChatPanel(Widget):
    """
    Right panel: a scrollable column of chat messages plus an input bar.

    Messages stream token-by-token into a live ChatMessage widget that is
    updated in place (Static.update), which avoids RichLog's inability to
    append inline text. The active LLM provider is shown in the border title.
    Supports /history to browse past sessions and `n` to start a new one.
    """

    DEFAULT_CSS = """
    ChatPanel {
        layout: vertical;
    }
    ChatPanel #chat-log {
        height: 1fr;
        padding: 0 1;
    }
    ChatPanel ChatMessage {
        margin: 0 0 1 0;
    }
    ChatPanel .msg-system {
        color: $text-muted;
    }
    ChatPanel #chat-input {
        dock: bottom;
        margin-top: 1;
    }
    """

    BINDINGS = [("n", "new_session", "New Chat")]

    def __init__(self, hardware_profile=None, **kwargs) -> None:
        super().__init__(**kwargs)
        self._hw = hardware_profile
        self._provider = None
        self._history = None
        self._system_prompt = ""
        self._is_streaming = False
        self._pending_sessions: list[dict] = []

    def compose(self) -> ComposeResult:
        yield VerticalScroll(id="chat-log")
        yield Input(placeholder="Type a message…  (/history · n=new session)", id="chat-input")

    def on_mount(self) -> None:
        from mlfit.tui.chat.base import resolve_provider
        from mlfit.tui.chat.history import ChatHistory

        self._provider = resolve_provider()
        hw_ctx = self._build_hw_context()
        self._history = ChatHistory.load_most_recent() or ChatHistory(hardware_context=hw_ctx)
        self._system_prompt = self._build_system_prompt()

        provider_name = type(self._provider).__name__.replace("Provider", "") if self._provider else "No LLM"
        self.border_title = f"Chat · {provider_name}"

        if self._provider is None:
            self._write_system(_NO_KEY_MSG)
            return

        for msg in self._history.messages:
            self._append_message(msg["role"], msg["content"])
        if not self._history.messages:
            self._write_system("Ask anything about model deployment on your hardware.")

    # --- rendering helpers -------------------------------------------------

    def _log(self) -> VerticalScroll:
        return self.query_one("#chat-log", VerticalScroll)

    def _append_message(self, role: str, content: str) -> ChatMessage:
        prefix = _USER_PREFIX if role == "user" else _ASSISTANT_PREFIX
        message = ChatMessage(role, prefix, content)
        self._log().mount(message)
        self._log().scroll_end(animate=False)
        return message

    def _write_system(self, text: str) -> None:
        note = Static(text)
        note.add_class("msg-system")
        self._log().mount(note)
        self._log().scroll_end(animate=False)

    # --- input handling ----------------------------------------------------

    def on_key(self, event) -> None:
        if event.key == "escape" and self.query_one("#chat-input", Input).has_focus:
            self._focus_neutral()
            event.stop()

    def _focus_neutral(self) -> None:
        """Move focus off the chat input so single-key shortcuts work again."""
        try:
            self.screen.query_one("#model-list").focus()
        except Exception:
            self.query_one("#chat-input", Input).blur()

    def on_input_submitted(self, event: Input.Submitted) -> None:
        text = event.value.strip()
        event.input.value = ""
        if not text:
            return

        if text == "/history":
            self._show_history()
            return

        if text.isdigit() and self._pending_sessions:
            self.load_session_number(int(text))
            return

        if self._provider is None:
            self._write_system("Set GEMINI_API_KEY to enable chat.")
            return

        if self._is_streaming:
            return

        self._send_message(text)

    def action_new_session(self) -> None:
        """Start a fresh chat session, discarding the current conversation."""
        from mlfit.tui.chat.history import ChatHistory

        self._history = ChatHistory(hardware_context=self._build_hw_context())
        self._system_prompt = self._build_system_prompt()
        self._pending_sessions = []
        self._log().remove_children()
        self._write_system("New session started. Ask anything about model deployment on your hardware.")

    # --- streaming ---------------------------------------------------------

    def _send_message(self, text: str) -> None:
        self._append_message("user", text)
        self._history.add("user", text)
        self.app.run_worker(self._stream_reply(), exclusive=False)

    async def _stream_reply(self) -> None:
        self._is_streaming = True
        message = self._append_message("assistant", "…")
        buffer = ""
        try:
            async for chunk in self._provider.send_message(
                messages=self._history.messages,
                system_prompt=self._system_prompt,
            ):
                buffer += chunk
                message.update(f"{_ASSISTANT_PREFIX}  {escape(buffer)}")
                self._log().scroll_end(animate=False)
        except Exception as exc:
            message.update(f"{_ASSISTANT_PREFIX}  [$error]Error: {escape(str(exc))}[/]")
        finally:
            if buffer:
                self._history.add("assistant", buffer)
            elif not buffer:
                message.update(f"{_ASSISTANT_PREFIX}  [$text-muted](no response)[/]")
            self._is_streaming = False

    # --- history browsing --------------------------------------------------

    def _show_history(self) -> None:
        from mlfit.tui.chat.history import ChatHistory

        sessions = ChatHistory.list_sessions()
        if not sessions:
            self._write_system("No past sessions found.")
            return

        self._pending_sessions = sessions
        lines = ["[bold]Past sessions[/] — type a number to load:"]
        for i, s in enumerate(sessions[:10], 1):
            ts = s["started_at"][:10] if s["started_at"] else "?"
            lines.append(f"  [$accent]{i}[/]. {ts}  ·  {s['message_count']} messages")
        self._write_system("\n".join(lines))

    def load_session_number(self, number: int) -> None:
        """Load a past session by its 1-indexed position from the /history list."""
        sessions = self._pending_sessions
        if not sessions or number < 1 or number > len(sessions):
            return
        from mlfit.tui.chat.history import ChatHistory

        self._history = ChatHistory.load(sessions[number - 1]["path"])
        self._pending_sessions = []
        self._log().remove_children()
        for msg in self._history.messages:
            self._append_message(msg["role"], msg["content"])
        self._write_system("— Session loaded —")

    # --- context / prompt --------------------------------------------------

    def update_context(self, model_id: str, recommend_result=None) -> None:
        """Update the system prompt when a model is selected in the browser."""
        self._system_prompt = self._build_system_prompt(model_id, recommend_result)

    def _build_hw_context(self) -> dict:
        if self._hw is None:
            return {}
        return {
            "gpu_backend": self._hw.gpu_backend,
            "total_vram_gb": self._hw.total_vram_gb,
            "ram_gb": self._hw.ram_gb,
        }

    def _build_system_prompt(self, model_id: str = "", recommend_result=None) -> str:
        hw_summary = "Unknown hardware"
        if self._hw:
            hw = self._hw
            if hw.gpus:
                gpu = hw.gpus[0]
                hw_summary = f"{gpu.name} ({gpu.vram_total_gb:.0f} GB VRAM), {hw.ram_gb:.0f} GB RAM, {hw.gpu_backend.upper()}"
            else:
                hw_summary = f"CPU-only, {hw.ram_gb:.0f} GB RAM"

        rec_summary = ""
        if recommend_result and recommend_result.best:
            best = recommend_result.best
            rec_summary = f"\nFor {model_id}, the top recommendation is {best.backend}."

        return (
            "You are an expert ML deployment advisor embedded in mlfit. "
            f"The user's hardware: {hw_summary}.{rec_summary} "
            "Answer questions about model deployment, backend selection, quantization, "
            "and performance optimization. Be concise and practical."
        )
