from __future__ import annotations

import csv
import json
import re
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

from nltk.stem import SnowballStemmer
from nltk.tokenize import TweetTokenizer
from spacy.lang.pt.stop_words import STOP_WORDS as SPACY_PT_STOP_WORDS


ENTREGA_1 = "entrega_1"
ENTREGA_2 = "entrega_2"

TOKENIZADOR_NLTK = TweetTokenizer(preserve_case=True, reduce_len=False, strip_handles=False)
RADICALIZADOR_NLTK = SnowballStemmer("portuguese")
REGEX_TOKEN_COM_CONTEUDO = re.compile(r"\w", re.UNICODE)
REGEX_URL = re.compile(r"https?://\S+|www\.\S+", re.IGNORECASE)
REGEX_MENCAO = re.compile(r"@\w+", re.UNICODE)
REGEX_HASHTAG = re.compile(r"#(\w+)", re.UNICODE)
REGEX_NAO_ALFANUMERICO = re.compile(r"[^\w\s]", re.UNICODE)
REGEX_ESPACOS = re.compile(r"\s+", re.UNICODE)


@dataclass(slots=True, frozen=True)
class ConfiguracaoEntrega:
    caminho_entrada: Path = Path("data/tweets.csv")
    pasta_saida_entrega_1: Path = Path("entregas/p1")
    pasta_saida_entrega_2: Path = Path("entregas/p2")

    def pasta_saida(self, tipo_entrega: str) -> Path:
        if tipo_entrega == ENTREGA_1:
            return self.pasta_saida_entrega_1
        if tipo_entrega == ENTREGA_2:
            return self.pasta_saida_entrega_2
        raise ValueError(f"Entrega desconhecida: {tipo_entrega}")

    def caminho_entrada_processamento(self, tipo_entrega: str) -> Path:
        if tipo_entrega == ENTREGA_1:
            return self.caminho_entrada
        if tipo_entrega == ENTREGA_2:
            return self.caminho_saida(ENTREGA_1)
        raise ValueError(f"Entrega desconhecida: {tipo_entrega}")

    def caminho_saida(self, tipo_entrega: str) -> Path:
        if tipo_entrega == ENTREGA_1:
            return self.pasta_saida(ENTREGA_1) / "entrega_1.csv"
        if tipo_entrega == ENTREGA_2:
            return self.pasta_saida(ENTREGA_2) / "entrega_2.csv"
        raise ValueError(f"Entrega desconhecida: {tipo_entrega}")

    def caminho_metadados(self, tipo_entrega: str) -> Path | None:
        if tipo_entrega == ENTREGA_2:
            return self.pasta_saida(ENTREGA_2) / "entrega_2_tfidf_features.json"
        return None


class ProcessadorEntrega:
    def __init__(
        self,
        tipo_entrega: str,
        callback_progresso: Callable[[int], None] | None = None,
        callback_status: Callable[[str], None] | None = None,
    ):
        if tipo_entrega not in {ENTREGA_1, ENTREGA_2}:
            raise ValueError(f"Entrega desconhecida: {tipo_entrega}")

        self.tipo_entrega = tipo_entrega
        self.callback_progresso = callback_progresso
        self.callback_status = callback_status

    def gerar(self) -> dict[str, int | str]:
        configuracao = ConfiguracaoEntrega()

        if not configuracao.caminho_entrada.exists():
            raise ValueError(f"CSV nao encontrado em {configuracao.caminho_entrada}.")

        configuracao.pasta_saida(self.tipo_entrega).mkdir(parents=True, exist_ok=True)
        caminho_entrada = self._garantir_entrada_processamento(configuracao)
        caminho_saida = configuracao.caminho_saida(self.tipo_entrega)

        self._emitir_status(
            f"Copiando {caminho_entrada} para {caminho_saida}..."
        )
        self._emitir_progresso(0)
        shutil.copyfile(caminho_entrada, caminho_saida)

        linhas, cabecalhos = self._ler_linhas(caminho_saida)
        if not linhas:
            raise ValueError("Nao ha linhas no CSV para processar.")

        caminho_metadados = ""

        if self.tipo_entrega == ENTREGA_1:
            novas_colunas = self._processar_entrega_1(linhas)
        else:
            novas_colunas, caminho_metadados = self._processar_entrega_2(
                linhas,
                configuracao,
            )

        cabecalhos_saida = [*cabecalhos, *novas_colunas]
        self._escrever_linhas(caminho_saida, cabecalhos_saida, linhas)
        self._emitir_progresso(100)

        return {
            "linhas": len(linhas),
            "caminho": str(caminho_saida),
            "caminho_metadados": caminho_metadados,
        }

    def _garantir_entrada_processamento(self, configuracao: ConfiguracaoEntrega) -> Path:
        caminho_entrada = configuracao.caminho_entrada_processamento(self.tipo_entrega)
        if self.tipo_entrega != ENTREGA_2:
            return caminho_entrada

        if caminho_entrada.exists():
            return caminho_entrada

        self._emitir_status(
            "Entrega 2 depende de entrega_1.csv. Gerando entrega 1 antes de continuar..."
        )

        ProcessadorEntrega(
            ENTREGA_1,
            callback_progresso=self.callback_progresso,
            callback_status=self.callback_status,
        ).gerar()
        return caminho_entrada

    def _processar_entrega_1(self, linhas: list[dict[str, str]]) -> list[str]:
        from processadores.entrega_1 import processar_entrega_1

        return processar_entrega_1(
            linhas=linhas,
            emitir_progresso=self._emitir_progresso,
            emitir_status=self._emitir_status,
        )

    def _processar_entrega_2(
        self,
        linhas: list[dict[str, str]],
        configuracao: ConfiguracaoEntrega,
    ) -> tuple[list[str], str]:
        from processadores.entrega_2 import processar_entrega_2

        return processar_entrega_2(
            linhas=linhas,
            configuracao=configuracao,
            emitir_progresso=self._emitir_progresso,
            emitir_status=self._emitir_status,
        )

    def _ler_linhas(self, caminho_csv: Path) -> tuple[list[dict[str, str]], list[str]]:
        with caminho_csv.open("r", newline="", encoding="utf-8") as arquivo_csv:
            leitor = csv.DictReader(arquivo_csv)
            linhas = list(leitor)
            return linhas, list(leitor.fieldnames or [])

    def _escrever_linhas(
        self,
        caminho_csv: Path,
        cabecalhos: list[str],
        linhas: list[dict[str, str]],
    ) -> None:
        with caminho_csv.open("w", newline="", encoding="utf-8") as arquivo_csv:
            escritor = csv.DictWriter(arquivo_csv, fieldnames=cabecalhos)
            escritor.writeheader()
            escritor.writerows(linhas)

    def _emitir_progresso(self, valor: int) -> None:
        if self.callback_progresso:
            self.callback_progresso(valor)

    def _emitir_status(self, mensagem: str) -> None:
        if self.callback_status:
            self.callback_status(mensagem)


def normalizar_texto_csv(valor: object) -> str:
    if valor is None:
        return ""
    return str(valor)

TEXTO_COM_EMOJIS = re.compile(
    r"[^a-zA-ZÀ-ÿ0-9\s.,!?'\-"
    r"#@"
    r"\U0001F300-\U0001F5FF"
    r"\U0001F600-\U0001F64F"
    r"\U0001F680-\U0001F6FF"
    r"\U0001F700-\U0001F77F"
    r"\U0001F780-\U0001F7FF"
    r"\U0001F800-\U0001F8FF"
    r"\U0001F900-\U0001F9FF"
    r"\U0001FA00-\U0001FAFF"
    r"\U00002700-\U000027BF"
    r"\U00002600-\U000026FF"
    r"\u200d\uFE0F]"
) 

TOKENS_NUMERICOS = re.compile(r"^\d+$", re.UNICODE)

def remover_decoracoes_com_regex(texto: str) -> str:
    # ☆．。．:*･ﾟ, ｡･:*:･ﾟ'☆
    return TEXTO_COM_EMOJIS.sub(" ", texto)

def remover_numericos_com_regex(texto: str) -> str:
    return " ".join(
        token
        for token in texto.split()
        if not TOKENS_NUMERICOS.match(token)
    )

def tokenizar_com_nltk(texto: str) -> list[str]:
    return [
        token
        for token in TOKENIZADOR_NLTK.tokenize(texto)
        if REGEX_TOKEN_COM_CONTEUDO.search(token)
    ]


def remover_stopwords_com_spacy(tokens: list[str]) -> list[str]:
    return [
        token
        for token in tokens
        if token.casefold() not in SPACY_PT_STOP_WORDS
    ]


def aplicar_stemming_com_nltk(tokens: list[str]) -> str:
    return " ".join(RADICALIZADOR_NLTK.stem(token.casefold()) for token in tokens)


def normalizar_com_regex(texto: str) -> str:
    texto_normalizado = texto.casefold()
    texto_normalizado = REGEX_URL.sub(" ", texto_normalizado)
    texto_normalizado = REGEX_HASHTAG.sub(r" \1 ", texto_normalizado)
    texto_normalizado = REGEX_MENCAO.sub(" ", texto_normalizado)
    texto_normalizado = REGEX_NAO_ALFANUMERICO.sub(" ", texto_normalizado)
    return REGEX_ESPACOS.sub(" ", texto_normalizado).strip()


def para_json(valor: object, ordenar_chaves: bool = False) -> str:
    return json.dumps(valor, ensure_ascii=False, sort_keys=ordenar_chaves)
