from textual.app import ComposeResult
from textual.widget import Widget

from mlfit.tui.widgets.model_browser import ModelBrowser, ModelSelected
from mlfit.tui.widgets.model_detail import ModelDetail, BackToBrowser
from mlfit.tui.model_meta import ModelRow


class LeftPanel(Widget):
    """
    Left column of the TUI: columnar model table or model detail view.

    Toggles between table and detail when a model is selected or the user
    presses Esc. Hardware info has moved to the app-level HardwareBar.
    """

    def __init__(self, hardware_profile=None, **kwargs) -> None:
        super().__init__(**kwargs)
        self._hw = hardware_profile

    def compose(self) -> ComposeResult:
        yield ModelBrowser(id="model-browser")
        detail = ModelDetail(id="model-detail")
        detail.display = False
        yield detail

    def on_model_selected(self, event: ModelSelected) -> None:
        event.stop()
        self._show_detail(event.model_id)

    def on_back_to_browser(self, event: BackToBrowser) -> None:
        event.stop()
        self._show_browser()

    def _show_detail(self, model_id: str) -> None:
        self.query_one("#model-browser", ModelBrowser).display = False
        detail = self.query_one("#model-detail", ModelDetail)
        detail.display = True
        detail.update_result(None)
        detail.focus()  # so Esc / b reach the detail view's bindings
        self.app.run_worker(self._load_and_show(model_id), exclusive=True)

    async def _load_and_show(self, model_id: str) -> None:
        import asyncio
        from mlfit.core.advisor import Advisor

        result = None
        try:
            result = await asyncio.to_thread(
                lambda: Advisor(model_id, gpuid=getattr(self.app, "gpuid", "all")).recommend()
            )
        except Exception as exc:
            self.app.notify(f"Error loading {model_id}: {exc}", severity="error")

        self.query_one("#model-detail", ModelDetail).update_result(result)
        self.app.post_message_to_chat(model_id, result)

        # A successful load may have fetched + cached a previously unknown model;
        # refresh the table so it appears as a normal row.
        if result is not None:
            self.app.refresh_models_from_cache()

    def _show_browser(self) -> None:
        self.query_one("#model-browser", ModelBrowser).display = True
        self.query_one("#model-detail", ModelDetail).display = False
        self.query_one("#model-list").focus()  # restore normal-mode navigation

    def set_models(self, rows: list[ModelRow]) -> None:
        """Replace the table rows after the cache is loaded or refreshed."""
        self.query_one("#model-browser", ModelBrowser).set_models(rows)
