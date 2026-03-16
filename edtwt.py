
from textual.app import App, ComposeResult
from textual.widgets import Footer, Header, Button
from textual.containers import Vertical
from extrator.anonymizer_interface import AnonymizerScreen
from extrator.interface import ExtratorScreen


class SelecionadorEtapa(App):
    BINDINGS = [
        ("d", "toggle_dark", "Toggle dark mode"),
        ("q", "quit", "Quit"),
    ]

    def compose(self) -> ComposeResult:
        yield Header()
        yield Vertical(
            Button("Abrir extrator", id="open_extrator"),
            Button("Exportar dataset anonimizado", id="open_anonymizer"),
        )
        yield Footer()

    def on_button_pressed(self, event: Button.Pressed) -> None:
        if event.button.id == "open_extrator":
            self.push_screen(ExtratorScreen())
        elif event.button.id == "open_anonymizer":
            self.push_screen(AnonymizerScreen())

    def action_toggle_dark(self) -> None:
        self.theme = (
            "textual-dark" if self.theme == "textual-light" else "textual-light"
        )

    async def action_quit(self) -> None:
        self.exit()


if __name__ == "__main__":
    app = SelecionadorEtapa()
    app.run()
