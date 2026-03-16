from textual.app import ComposeResult
from textual.containers import Vertical
from textual.screen import Screen
from textual.widgets import Button, ProgressBar, Static

from extrator.anonymizer import Anonymizer
from extrator.extraction_progress import ExtractionProgress


class AnonymizerScreen(Screen):
    BINDINGS = [("q", "pop_screen", "Voltar")]

    def compose(self) -> ComposeResult:
        yield Vertical(
            Button("Exportar dataset anonimizado", id="start_btn"),
            Static("Exporte uma versao anonimizada do banco SQLite.", id="status_display"),
            ExtractionProgress(id="progress_display"),
            ProgressBar(total=100, id="progress_bar"),
        )

    def on_button_pressed(self, event: Button.Pressed) -> None:
        if event.button.id == "start_btn":
            self.start_export()

    def start_export(self) -> None:
        start_button = self.query_one("#start_btn", Button)
        start_button.disabled = True
        self.update_progress(0)
        self.update_status("Preparando exportacao anonimizada...")

        anonymizer = Anonymizer(
            progress_callback=lambda value: self.app.call_from_thread(self.update_progress, value),
            status_callback=lambda message: self.app.call_from_thread(self.update_status, message),
        )

        def worker() -> None:
            try:
                result = anonymizer.export()
                self.app.call_from_thread(
                    self.finish_export,
                    f"Exportacao concluida: {result['rows']} linha(s) em {result['path']}.",
                )
            except Exception as exc:
                self.app.call_from_thread(
                    self.finish_export,
                    f"Falha na exportacao: {exc}",
                    True,
                )

        self.run_worker(worker, thread=True)

    def finish_export(self, message: str, failed: bool = False) -> None:
        self.query_one("#start_btn", Button).disabled = False
        self.update_status(message)
        if failed:
            self.update_progress(0)

    def update_progress(self, value: int) -> None:
        progress_display = self.query_one("#progress_display", ExtractionProgress)
        progress_bar = self.query_one("#progress_bar", ProgressBar)
        progress_display.set_progress(value)
        progress_bar.progress = value

    def update_status(self, message: str) -> None:
        self.query_one("#status_display", Static).update(message)

    def action_pop_screen(self) -> None:
        self.app.pop_screen()
