from datetime import date
from threading import Event

from textual.app import ComposeResult
from textual.screen import Screen
from textual.widgets import Button, Input, ProgressBar, Static
from textual.containers import Vertical

from extrator.extraction_progress import ExtractionProgress
from extrator.extrator import Extrator, SEARCH_BY_DATE, SEARCH_TOPS


class ExtratorScreen(Screen):
    selected_search_mode = SEARCH_BY_DATE
    top_prompt_response = False

    BINDINGS = [
        ("q", "pop_screen", "Voltar"),
    ]

    def compose(self) -> ComposeResult:
        yield Vertical(
            Static("Modo de busca", id="search_mode_label"),
            Button("Buscar por data (selecionado)", id="mode_date_btn"),
            Button("Buscar por tops", id="mode_top_btn"),
            Static("Data mais recente (YYYY-MM-DD, opcional)", id="newest_date_label"),
            Input(placeholder="2026-03-15", id="newest_date_input"),
            Static("Data mais antiga (YYYY-MM-DD, opcional)", id="oldest_date_label"),
            Input(placeholder="2026-02-01", id="oldest_date_input"),
            Button("Iniciar extração", id="start_btn"),
            Button("Parar após a raspagem atual", id="stop_btn", disabled=True),
            Button("Buscar mais 10 páginas", id="continue_top_btn", disabled=True),
            Button("Parar paginação Top", id="stop_top_btn", disabled=True),
            Static("Configure o .env e inicie a extração.", id="status_display"),
            ExtractionProgress(id="progress_display"),
            ProgressBar(total=100, id="progress_bar"),
        )

    def on_button_pressed(self, event: Button.Pressed) -> None:
        if event.button.id == "mode_date_btn":
            self.set_search_mode(SEARCH_BY_DATE)
        elif event.button.id == "mode_top_btn":
            self.set_search_mode(SEARCH_TOPS)
        elif event.button.id == "start_btn":
            self.start_extraction()
        elif event.button.id == "stop_btn":
            self.stop_extraction()
        elif event.button.id == "continue_top_btn":
            self.respond_top_prompt(True)
        elif event.button.id == "stop_top_btn":
            self.respond_top_prompt(False)

    def start_extraction(self) -> None:
        start_button = self.query_one("#start_btn", Button)
        stop_button = self.query_one("#stop_btn", Button)
        newest_date_input = self.query_one("#newest_date_input", Input)
        oldest_date_input = self.query_one("#oldest_date_input", Input)

        try:
            newest_date = self.parse_optional_date(newest_date_input.value)
            oldest_date = self.parse_optional_date(oldest_date_input.value)
        except ValueError as exc:
            self.update_status(str(exc))
            self.update_progress(0)
            return

        if self.selected_search_mode == SEARCH_TOPS:
            newest_date = None
            oldest_date = None

        start_button.disabled = True
        stop_button.disabled = False
        newest_date_input.disabled = True
        oldest_date_input.disabled = True
        self.query_one("#mode_date_btn", Button).disabled = True
        self.query_one("#mode_top_btn", Button).disabled = True
        self.update_progress(0)
        self.update_status("Preparando extração...")

        self.extrator = Extrator(
            "data/tweets.csv",
            progress_callback=lambda value: self.app.call_from_thread(self.update_progress, value),
            status_callback=lambda message: self.app.call_from_thread(self.update_status, message),
            top_batch_prompt_callback=self.prompt_top_batch_continuation,
        )

        def worker() -> None:
            try:
                result = self.extrator.extrair(
                    newest_date=newest_date,
                    oldest_date=oldest_date,
                    search_mode=self.selected_search_mode,
                )
                if result.get("search_mode") == SEARCH_TOPS:
                    final_message = (
                        f"Busca Top encerrada: {result['rows']} tweet(s) salvos. "
                        f"CSV: {result['path']} | SQLite: {result['database_path']}."
                    )
                else:
                    final_message = (
                        f"Extração encerrada: {result['rows']} tweet(s) em {result['dates']} dia(s), "
                        f"intervalo {result['start_date']} -> {result['end_date']}. "
                        f"CSV: {result['path']} | SQLite: {result['database_path']}."
                    )

                if result.get("stopped"):
                    final_message = (
                        f"Extração parada pelo usuário: {result['rows']} tweet(s) em {result['dates']} dia(s), "
                        f"intervalo {result['start_date']} -> {result['end_date']}. "
                        f"CSV: {result['path']} | SQLite: {result['database_path']}."
                    )

                self.app.call_from_thread(
                    self.finish_extraction,
                    final_message,
                )
            except Exception as exc:
                self.app.call_from_thread(
                    self.finish_extraction,
                    f"Falha na extração: {exc}",
                    True,
                )

        self.run_worker(worker, thread=True)

    def stop_extraction(self) -> None:
        stop_button = self.query_one("#stop_btn", Button)
        stop_button.disabled = True
        self.update_status("Parada solicitada. O extrator vai encerrar após concluir a raspagem atual.")

        if hasattr(self, "extrator"):
            self.extrator.solicitar_parada()

        if hasattr(self, "top_prompt_event"):
            self.top_prompt_response = False
            self.top_prompt_event.set()

    def finish_extraction(self, message: str, failed: bool = False) -> None:
        start_button = self.query_one("#start_btn", Button)
        stop_button = self.query_one("#stop_btn", Button)
        newest_date_input = self.query_one("#newest_date_input", Input)
        oldest_date_input = self.query_one("#oldest_date_input", Input)
        start_button.disabled = False
        stop_button.disabled = True
        newest_date_input.disabled = False
        oldest_date_input.disabled = False
        self.query_one("#mode_date_btn", Button).disabled = False
        self.query_one("#mode_top_btn", Button).disabled = False
        self.query_one("#continue_top_btn", Button).disabled = True
        self.query_one("#stop_top_btn", Button).disabled = True
        self.set_search_mode(self.selected_search_mode)
        self.update_status(message)

        if failed:
            self.update_progress(0)

    def parse_optional_date(self, value: str) -> date | None:
        normalized_value = value.strip()
        if not normalized_value:
            return None

        try:
            return date.fromisoformat(normalized_value)
        except ValueError as exc:
            raise ValueError(
                "Use datas no formato YYYY-MM-DD nos campos de intervalo."
            ) from exc

    def set_search_mode(self, mode: str) -> None:
        self.selected_search_mode = mode
        newest_date_input = self.query_one("#newest_date_input", Input)
        oldest_date_input = self.query_one("#oldest_date_input", Input)
        mode_date_btn = self.query_one("#mode_date_btn", Button)
        mode_top_btn = self.query_one("#mode_top_btn", Button)

        is_top_mode = mode == SEARCH_TOPS
        newest_date_input.disabled = is_top_mode
        oldest_date_input.disabled = is_top_mode
        mode_date_btn.label = "Buscar por data (selecionado)" if not is_top_mode else "Buscar por data"
        mode_top_btn.label = "Buscar por tops (selecionado)" if is_top_mode else "Buscar por tops"

        if is_top_mode:
            self.update_status("Modo Top selecionado. A busca ignora datas e usa queryType=Top.")
        else:
            self.update_status("Modo Data selecionado. A busca usa intervalo de datas com queryType=Latest.")

    def prompt_top_batch_continuation(self, page_number: int) -> bool:
        self.top_prompt_event = Event()
        self.top_prompt_response = False
        self.app.call_from_thread(self.show_top_prompt, page_number)
        self.top_prompt_event.wait()
        return self.top_prompt_response

    def show_top_prompt(self, page_number: int) -> None:
        self.query_one("#continue_top_btn", Button).disabled = False
        self.query_one("#stop_top_btn", Button).disabled = False
        self.update_status(
            f"{page_number} páginas Top foram buscadas. Deseja buscar mais 10 páginas?"
        )

    def respond_top_prompt(self, should_continue: bool) -> None:
        self.query_one("#continue_top_btn", Button).disabled = True
        self.query_one("#stop_top_btn", Button).disabled = True
        self.top_prompt_response = should_continue
        if hasattr(self, "top_prompt_event"):
            self.top_prompt_event.set()

    def update_progress(self, value: int) -> None:
        progress_display = self.query_one("#progress_display", ExtractionProgress)
        progress_bar = self.query_one("#progress_bar", ProgressBar)
        progress_display.set_progress(value)
        progress_bar.progress = value

    def update_status(self, message: str) -> None:
        status_display = self.query_one("#status_display", Static)
        status_display.update(message)

    def action_pop_screen(self) -> None:
        self.app.pop_screen()
