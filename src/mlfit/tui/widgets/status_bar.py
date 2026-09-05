from rich.text import Text
from textual.widgets import Static


class StatusBar(Static):
    """
    Bottom bar showing keyboard shortcut hints on a solid, filled background.

    Rendered from a Rich Text object (not markup) so the literal ``[l]`` style
    key hints are shown verbatim rather than parsed as markup tags.
    """

    DEFAULT_CSS = """
    StatusBar {
        height: 1;
        background: $panel;
        color: $text-muted;
        text-align: center;
        content-align: center middle;
    }
    """

    _HINTS = (
        "[jk/↑↓] Nav   [/] Search   [↵] Select   [i] Message   [l] Left   "
        "[c] Chat   [t] Theme   [h] Help   [q] Quit"
    )

    def on_mount(self) -> None:
        self.update(Text(self._HINTS))
