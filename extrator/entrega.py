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
from sklearn.feature_extraction.text import TfidfVectorizer
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
    pasta_saida: Path = Path("entregas/p1")

    def caminho_saida(self, tipo_entrega: str) -> Path:
        if tipo_entrega == ENTREGA_1:
            return self.pasta_saida / "entrega_1.csv"
        if tipo_entrega == ENTREGA_2:
            return self.pasta_saida / "entrega_2.csv"
        raise ValueError(f"Entrega desconhecida: {tipo_entrega}")

    def caminho_metadados(self, tipo_entrega: str) -> Path | None:
        if tipo_entrega == ENTREGA_2:
            return self.pasta_saida / "entrega_2_tfidf_features.json"
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

        configuracao.pasta_saida.mkdir(parents=True, exist_ok=True)
        caminho_saida = configuracao.caminho_saida(self.tipo_entrega)

        self._emitir_status(
            f"Copiando {configuracao.caminho_entrada} para {caminho_saida}..."
        )
        self._emitir_progresso(0)
        shutil.copyfile(configuracao.caminho_entrada, caminho_saida)

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

    def _processar_entrega_1(self, linhas: list[dict[str, str]]) -> list[str]:
        self._emitir_status(
            "Entrega 1: tokenizacao com NLTK, remocao de stopwords com spaCy e stemming com NLTK..."
        )

        total_linhas = len(linhas)
        for indice, linha in enumerate(linhas, start=1):
            texto = normalizar_texto_csv(linha.get("text"))
            tokens = tokenizar_com_nltk(texto)
            tokens_sem_stopwords = remover_stopwords_com_spacy(tokens)
            texto_radicalizado = aplicar_stemming_com_nltk(tokens_sem_stopwords)

            linha["tokenizacao_nltk"] = para_json(tokens)
            linha["remocao_stopwords_spacy"] = para_json(tokens_sem_stopwords)
            linha["stemming_nltk"] = texto_radicalizado

            self._emitir_progresso(int(indice / total_linhas * 100))

        return [
            "tokenizacao_nltk",
            "remocao_stopwords_spacy",
            "stemming_nltk",
        ]

    def _processar_entrega_2(
        self,
        linhas: list[dict[str, str]],
        configuracao: ConfiguracaoEntrega,
    ) -> tuple[list[str], str]:
        self._emitir_status(
            "Entrega 2: normalizacao com re e selecao de features com scikit-learn..."
        )

        total_linhas = len(linhas)
        textos_normalizados: list[str] = []

        for indice, linha in enumerate(linhas, start=1):
            texto_normalizado = normalizar_com_regex(normalizar_texto_csv(linha.get("text")))
            linha["normalizacao_re"] = texto_normalizado
            textos_normalizados.append(texto_normalizado)
            self._emitir_progresso(int(indice / total_linhas * 50))

        nomes_features: list[str] = []
        features_por_linha: list[str] = [para_json({}) for _ in linhas]

        if any(textos_normalizados):
            vetorizador = TfidfVectorizer(
                lowercase=False,
                max_df=0.85,
                max_features=50,
                token_pattern=r"(?u)\b\w+\b",
            )

            try:
                matriz = vetorizador.fit_transform(textos_normalizados)
                nomes_features = vetorizador.get_feature_names_out().tolist()

                for indice, vetor_linha in enumerate(matriz, start=1):
                    mapa_features = {
                        nomes_features[indice_coluna]: round(float(valor), 6)
                        for indice_coluna, valor in zip(vetor_linha.indices, vetor_linha.data)
                    }
                    features_por_linha[indice - 1] = para_json(
                        mapa_features,
                        ordenar_chaves=True,
                    )
                    self._emitir_progresso(50 + int(indice / total_linhas * 50))
            except ValueError:
                self._emitir_progresso(100)
        else:
            self._emitir_progresso(100)

        for linha, mapa_features in zip(linhas, features_por_linha):
            linha["features_tfidf_sklearn"] = mapa_features

        caminho_metadados = configuracao.caminho_metadados(self.tipo_entrega)
        if caminho_metadados is not None:
            caminho_metadados.write_text(
                json.dumps(
                    {
                        "feature_names": nomes_features,
                        "vectorizer": {
                            "max_df": 0.85,
                            "max_features": 50,
                            "token_pattern": r"(?u)\b\w+\b",
                        },
                    },
                    ensure_ascii=False,
                    indent=2,
                ),
                encoding="utf-8",
            )

        return ["normalizacao_re", "features_tfidf_sklearn"], str(caminho_metadados or "")

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
