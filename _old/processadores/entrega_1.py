from __future__ import annotations

from typing import Callable

from extrator.entrega import (
    aplicar_stemming_com_nltk,
    normalizar_texto_csv,
    para_json,
    remover_decoracoes_com_regex,
    remover_numericos_com_regex,
    remover_stopwords_com_spacy,
    tokenizar_com_nltk,
)


COLUNAS_ENTREGA_1 = [
    "tokenizacao_nltk",
    "remocao_stopwords_spacy",
    "stemming_nltk",
]


def processar_entrega_1(
    linhas: list[dict[str, str]],
    emitir_progresso: Callable[[int], None],
    emitir_status: Callable[[str], None],
) -> list[str]:
    emitir_status(
        "Entrega 1: tokenizacao com NLTK, remocao de stopwords com spaCy e stemming com NLTK..."
    )

    total_linhas = len(linhas)
    for indice, linha in enumerate(linhas, start=1):
        texto = normalizar_texto_csv(linha.get("text"))
        texto = remover_decoracoes_com_regex(texto)
        texto = remover_numericos_com_regex(texto)
        tokens = tokenizar_com_nltk(texto)
        tokens_sem_stopwords = remover_stopwords_com_spacy(tokens)
        texto_radicalizado = aplicar_stemming_com_nltk(tokens_sem_stopwords)

        linha["tokenizacao_nltk"] = para_json(tokens)
        linha["remocao_stopwords_spacy"] = para_json(tokens_sem_stopwords)
        linha["stemming_nltk"] = texto_radicalizado

        emitir_progresso(int(indice / total_linhas * 100))

    return COLUNAS_ENTREGA_1.copy()
