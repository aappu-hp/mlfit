import logging
import pathlib

logger = logging.getLogger(__name__)

_THEME_FILE = pathlib.Path.home() / ".mlfit" / "theme"

DEFAULT_THEME = "nord"

CYCLE_THEMES: tuple[str, ...] = (
    "nord",
    "gruvbox",
    "dracula",
    "monokai",
    "solarized-dark",
    "catppuccin-mocha",
    "catppuccin-macchiato",
    "tokyo-night",
    "rose-pine",
    "textual-dark",
    "textual-light",
    "catppuccin-latte",
    "solarized-light",
)


class ThemeManager:
    """
    Manages the active TUI theme: cycling through a curated list and
    persisting the choice to ~/.mlfit/theme so it survives restarts.

    All themes referenced here ship built-in with Textual, so no custom
    theme registration is required.
    """

    def __init__(self, themes: tuple[str, ...] = CYCLE_THEMES) -> None:
        self._themes = themes

    def load(self) -> str:
        """
        Return the saved theme name, or the default if none is saved or valid.

        Returns:
            A theme name guaranteed to be in the cycle list.
        """
        try:
            saved = _THEME_FILE.read_text().strip()
        except OSError:
            return DEFAULT_THEME
        return saved if saved in self._themes else DEFAULT_THEME

    def save(self, theme_name: str) -> None:
        """
        Persist the given theme name to disk.

        Args:
            theme_name: The Textual theme name to remember.
        """
        try:
            _THEME_FILE.parent.mkdir(parents=True, exist_ok=True)
            _THEME_FILE.write_text(theme_name)
        except OSError as exc:
            logger.warning("Could not save theme '%s': %s", theme_name, exc)

    def next_theme(self, current: str) -> str:
        """
        Return the next theme in the cycle after the current one.

        Args:
            current: The currently active theme name.

        Returns:
            The next theme name, wrapping around to the start.
        """
        try:
            index = self._themes.index(current)
        except ValueError:
            return self._themes[0]
        return self._themes[(index + 1) % len(self._themes)]
