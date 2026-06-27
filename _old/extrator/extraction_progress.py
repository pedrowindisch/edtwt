from textual.widgets import Static
from textual.reactive import reactive

class ExtractionProgress(Static):
    progress = reactive(0)

    def render(self) -> str:
        return f"Progresso: {self.progress}%"

    def set_progress(self, value: int):
        self.progress = value