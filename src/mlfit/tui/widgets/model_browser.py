from rich.text import Text

from textual.app import ComposeResult
from textual.message import Message
from textual.widget import Widget
from textual.widgets import Input, DataTable

from mlfit.tui.fit import FitLevel
from mlfit.tui.model_meta import ModelRow, fmt_ctx, fmt_params


class ModelSelected(Message):
    """Posted when the user selects a model (or the fetch row) in the browser."""

    def __init__(self, model_id: str) -> None:
        super().__init__()
        self.model_id = model_id


_FETCH_KEY = "__fetch__"

_FIT_COLORS: dict[FitLevel, tuple[str, str]] = {
    FitLevel.PERFECT: ("●", "green"),
    FitLevel.GOOD: ("●", "yellow"),
    FitLevel.MARGINAL: ("●", "magenta"),
    FitLevel.TIGHT: ("●", "red"),
    FitLevel.UNKNOWN: ("○", "grey50"),
}


class _ModelTable(DataTable):
    """DataTable with vim-style navigation (j/k, g/G)."""

    BINDINGS = [
        *DataTable.BINDINGS,
        ("j", "cursor_down", "Down"),
        ("k", "cursor_up", "Up"),
        ("g", "scroll_top", "Top"),
        ("G", "scroll_bottom", "Bottom"),
    ]

    def action_scroll_top(self) -> None:
        if self.row_count:
            self.move_cursor(row=0)

    def action_scroll_bottom(self) -> None:
        if self.row_count:
            self.move_cursor(row=self.row_count - 1)


class ModelBrowser(Widget):
    """
    Columnar, searchable table of cached models.

    Columns: Fit (colored dot) · Model · Provider · Params · Ctx. Rows are built
    by the app from the local cache (no network). When a search matches nothing
    but looks like a model id ("org/name"), a synthetic "fetch from HuggingFace"
    row is offered, whose selection triggers the cache-miss download path.

    Emits ModelSelected on Enter. Supports vim navigation and / to search.
    """

    DEFAULT_CSS = """
    ModelBrowser {
        height: 1fr;
    }
    ModelBrowser Input {
        margin-bottom: 1;
    }
    ModelBrowser _ModelTable {
        height: 1fr;
        border: none;
    }
    """

    BINDINGS = [("slash", "focus_search", "Search")]

    def __init__(self, model_ids: list[str] | None = None, **kwargs) -> None:
        super().__init__(**kwargs)
        self._rows: list[ModelRow] = []
        self._filtered: list[ModelRow] = []
        self._query: str = ""

    def compose(self) -> ComposeResult:
        yield Input(placeholder="/ to search or type org/model to fetch...", id="model-search")
        table = _ModelTable(id="model-list", cursor_type="row", zebra_stripes=False)
        table.add_column("Fit", key="fit", width=3)
        table.add_column("Model", key="model")
        table.add_column("Provider", key="provider")
        table.add_column("Params", key="params", width=8)
        table.add_column("Ctx", key="ctx", width=7)
        yield table

    def on_mount(self) -> None:
        # Focus the table (not search) so vim keys and Enter work immediately.
        self.query_one("#model-list", _ModelTable).focus()

    def action_focus_search(self) -> None:
        self.query_one("#model-search", Input).focus()

    def on_key(self, event) -> None:
        search = self.query_one("#model-search", Input)
        if not search.has_focus:
            return
        if event.key == "escape":
            self.query_one("#model-list", _ModelTable).focus()
            event.stop()
        elif event.key == "ctrl+u":
            search.value = ""
            event.stop()

    def on_input_changed(self, event: Input.Changed) -> None:
        self._query = event.value.strip()
        query = self._query.lower()
        self._filtered = [r for r in self._rows if query in r.model_id.lower()]
        self._rebuild_table()

    def on_data_table_row_selected(self, event: _ModelTable.RowSelected) -> None:
        key = event.row_key.value
        if key is None:
            return
        if key == _FETCH_KEY:
            if self._query:
                self.post_message(ModelSelected(self._query))
        else:
            self.post_message(ModelSelected(key))

    def set_models(self, rows: list[ModelRow]) -> None:
        """Replace all model rows (called by the app after cache load/refresh)."""
        self._rows = list(rows)
        query = self._query.lower()
        self._filtered = [r for r in self._rows if query in r.model_id.lower()]
        self._rebuild_table()

    def _rebuild_table(self) -> None:
        table = self.query_one("#model-list", _ModelTable)
        table.clear()

        if self._filtered:
            for row in self._filtered:
                table.add_row(
                    self._fit_cell(row.fit),
                    row.name,
                    row.provider,
                    fmt_params(row.params_b),
                    fmt_ctx(row.context),
                    key=row.model_id,
                )
            table.move_cursor(row=0)
        elif self._looks_like_model_id(self._query):
            table.add_row(
                Text("⤓", style="cyan"),
                Text(f'Fetch "{self._query}" from HuggingFace…', style="cyan"),
                "", "", "",
                key=_FETCH_KEY,
            )
            table.move_cursor(row=0)

    @staticmethod
    def _looks_like_model_id(query: str) -> bool:
        return "/" in query and len(query) > 3

    @staticmethod
    def _fit_cell(fit: FitLevel) -> Text:
        dot, color = _FIT_COLORS.get(fit, _FIT_COLORS[FitLevel.UNKNOWN])
        return Text(dot, style=color)
