from __future__ import annotations

from textual.app import ComposeResult
from textual.containers import Vertical
from textual.screen import Screen
from textual.widgets import Button, ProgressBar, Static

from extrator.entrega import ENTREGA_1, ENTREGA_2, ProcessadorEntrega
from extrator.extraction_progress import ExtractionProgress


class EntregaScreen(Screen):
    BINDINGS = [("q", "pop_screen", "Voltar")]

    def __init__(self, entrega_kind: str) -> None:
        super().__init__()
        self.entrega_kind = entrega_kind

    def compose(self) -> ComposeResult:
        yield Vertical(
            Button(self.rotulo_botao, id="start_btn"),
            Static(self.descricao_tela, id="status_display"),
            ExtractionProgress(id="progress_display"),
            ProgressBar(total=100, id="progress_bar"),
        )

    @property
    def rotulo_botao(self) -> str:
        if self.entrega_kind == ENTREGA_1:
            return "Gerar entrega 1"
        return "Gerar entrega 2"

    @property
    def descricao_tela(self) -> str:
        if self.entrega_kind == ENTREGA_1:
            return (
                "Copia data/tweets.csv e adiciona tokenizacao (NLTK), "
                "remocao de stopwords (spaCy) e stemming (NLTK)."
            )
        return (
            "Copia data/tweets.csv e adiciona normalizacao textual (re) "
            "e features TF-IDF (scikit-learn)."
        )

    def on_button_pressed(self, event: Button.Pressed) -> None:
        if event.button.id == "start_btn":
            self.iniciar_entrega()

    def iniciar_entrega(self) -> None:
        botao_iniciar = self.query_one("#start_btn", Button)
        botao_iniciar.disabled = True
        self.atualizar_progresso(0)
        self.atualizar_status(f"Preparando {self.rotulo_botao.lower()}...")

        processador = ProcessadorEntrega(
            self.entrega_kind,
            callback_progresso=lambda valor: self.app.call_from_thread(self.atualizar_progresso, valor),
            callback_status=lambda mensagem: self.app.call_from_thread(self.atualizar_status, mensagem),
        )

        def trabalhador() -> None:
            try:
                resultado = processador.gerar()
                mensagem = (
                    f"{self.rotulo_botao} concluida: {resultado['linhas']} linha(s) em {resultado['caminho']}."
                )
                if resultado["caminho_metadados"]:
                    mensagem += f" Metadados em {resultado['caminho_metadados']}."

                self.app.call_from_thread(self.finalizar_entrega, mensagem)
            except Exception as exc:
                self.app.call_from_thread(
                    self.finalizar_entrega,
                    f"Falha na entrega: {exc}",
                    True,
                )

        self.run_worker(trabalhador, thread=True)

    def finalizar_entrega(self, mensagem: str, falhou: bool = False) -> None:
        self.query_one("#start_btn", Button).disabled = False
        self.atualizar_status(mensagem)
        if falhou:
            self.atualizar_progresso(0)

    def atualizar_progresso(self, valor: int) -> None:
        exibicao_progresso = self.query_one("#progress_display", ExtractionProgress)
        barra_progresso = self.query_one("#progress_bar", ProgressBar)
        exibicao_progresso.set_progress(valor)
        barra_progresso.progress = valor

    def atualizar_status(self, mensagem: str) -> None:
        self.query_one("#status_display", Static).update(mensagem)

    def action_pop_screen(self) -> None:
        self.app.pop_screen()
