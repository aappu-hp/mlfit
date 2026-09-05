from textual.app import ComposeResult
from textual.containers import Grid, VerticalScroll
from textual.screen import ModalScreen
from textual.widgets import Static


KEY_BINDINGS: list[tuple[str, list[tuple[str, str]]]] = [
    ("Navigation", [
        ("↑ / k", "Move up"),
        ("↓ / j", "Move down"),
        ("g / G", "Jump to top / bottom"),
        ("Enter", "Select model · send chat message"),
        ("/", "Search models"),
        ("Ctrl-U", "Clear search"),
        ("Esc", "Leave input · back from detail"),
        ("Tab", "Switch focus (models ↔ chat)"),
    ]),
    ("Chat", [
        ("i", "Type a message (focus input)"),
        ("Enter", "Send message"),
        ("n", "New chat session"),
    ]),
    ("View & Panels", [
        ("l", "Toggle left panel"),
        ("c", "Toggle chat panel"),
        ("t", "Cycle theme"),
    ]),
    ("Actions", [
        ("h", "Toggle this help"),
        ("q", "Quit"),
    ]),
]


class HelpScreen(ModalScreen):
    """
    Centered modal overlay listing every keyboard shortcut, grouped by section.

    Rendered from KEY_BINDINGS so the overlay never drifts from the real
    bindings. Dismissed with Esc, h, or q. Mirrors llmfit's help popup.
    """

    BINDINGS = [
        ("escape", "dismiss", "Close"),
        ("h", "dismiss", "Close"),
        ("q", "dismiss", "Close"),
    ]

    DEFAULT_CSS = """
    HelpScreen {
        align: center middle;
    }
    HelpScreen #help-box {
        width: auto;
        max-width: 56;
        height: auto;
        max-height: 90%;
        padding: 1 2;
        background: $panel;
        border: round $primary;
    }
    HelpScreen .help-section {
        color: $primary;
        text-style: bold;
        margin: 1 0 0 0;
    }
    HelpScreen .help-grid {
        grid-size: 2;
        grid-columns: 11 34;
        grid-rows: auto;
        height: auto;
        margin: 0 0 0 1;
    }
    HelpScreen .help-key {
        color: $accent;
        text-style: bold;
    }
    HelpScreen .help-desc {
        color: $foreground;
    }
    HelpScreen #help-hint {
        margin: 1 0 0 0;
        color: $text-muted;
        text-align: center;
    }
    """

    def compose(self) -> ComposeResult:
        with VerticalScroll(id="help-box") as box:
            box.border_title = "Key Bindings"
            for section, bindings in KEY_BINDINGS:
                yield Static(section, classes="help-section")
                with Grid(classes="help-grid"):
                    for keys, desc in bindings:
                        yield Static(keys, classes="help-key")
                        yield Static(desc, classes="help-desc")
            yield Static("Esc · h · q  to close", id="help-hint")
