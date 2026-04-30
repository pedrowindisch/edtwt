from __future__ import annotations

import json
from typing import Callable

import numpy as np
from gensim.models import Word2Vec
from nltk.tokenize import TweetTokenizer

from extrator.entrega import ENTREGA_3, ConfiguracaoEntrega, normalizar_texto_csv, para_json


TOKENIZADOR = TweetTokenizer(preserve_case=False, reduce_len=True, strip_handles=False)

COLUNAS_ENTREGA_3 = [
    "tokens_word2vec",
    "vetor_medio_word2vec",
]

# Parâmetros do modelo Word2Vec (igual ao notebook do professor)
VECTOR_DIM  = 100
WINDOW_SIZE = 5
MIN_COUNT   = 1
NUM_WORKERS = 4


def _tokenizar(texto: str) -> list[str]:
    """Tokeniza e retorna apenas tokens com conteúdo alfabético."""
    return [t for t in TOKENIZADOR.tokenize(texto) if any(c.isalpha() for c in t)]


def processar_entrega_3(
    linhas: list[dict[str, str]],
    configuracao: ConfiguracaoEntrega,
    emitir_progresso: Callable[[int], None],
    emitir_status: Callable[[str], None],
) -> tuple[list[str], str]:

    emitir_status("Entrega 3: treinando Word2Vec com Gensim...")

    total_linhas = len(linhas)

    # ── 1. Tokenizar corpus ──────────────────────────────────────────────────
    corpus_tokenizado: list[list[str]] = []
    for indice, linha in enumerate(linhas, start=1):
        if "normalizacao_re" not in linha:
            raise ValueError(
                "Entrega 3 precisa das colunas geradas pela Entrega 2, incluindo normalizacao_re."
            )
        texto = normalizar_texto_csv(linha.get("normalizacao_re"))
        tokens = _tokenizar(texto)
        corpus_tokenizado.append(tokens)
        linha["tokens_word2vec"] = para_json(tokens)
        emitir_progresso(int(indice / total_linhas * 30))

    # ── 2. Treinar Word2Vec ──────────────────────────────────────────────────
    emitir_status("Entrega 3: treinando modelo Word2Vec (CBOW)...")
    modelo = Word2Vec(
        sentences=corpus_tokenizado,
        vector_size=VECTOR_DIM,
        window=WINDOW_SIZE,
        min_count=MIN_COUNT,
        workers=NUM_WORKERS,
        sg=0,  # CBOW (igual ao notebook do professor)
    )
    emitir_progresso(60)

    # ── 3. Calcular vetor médio por tweet ────────────────────────────────────
    emitir_status("Entrega 3: calculando vetores médios por tweet...")
    for indice, (linha, tokens) in enumerate(zip(linhas, corpus_tokenizado), start=1):
        vetores = [modelo.wv[t] for t in tokens if t in modelo.wv]
        if vetores:
            vetor_medio = np.mean(vetores, axis=0).tolist()
        else:
            vetor_medio = [0.0] * VECTOR_DIM
        linha["vetor_medio_word2vec"] = para_json(
            [round(v, 6) for v in vetor_medio]
        )
        emitir_progresso(60 + int(indice / total_linhas * 35))

    # ── 4. Salvar metadados do modelo ────────────────────────────────────────
    caminho_metadados = configuracao.caminho_metadados(ENTREGA_3)
    if caminho_metadados is not None:
        metadados = {
            "vocabulario_tamanho": len(modelo.wv.index_to_key),
            "vector_size": VECTOR_DIM,
            "window": WINDOW_SIZE,
            "min_count": MIN_COUNT,
            "sg": 0,
            "arquitetura": "CBOW",
            "vocabulario_amostra": modelo.wv.index_to_key[:50],
        }
        caminho_metadados.write_text(
            json.dumps(metadados, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

    emitir_progresso(100)
    return COLUNAS_ENTREGA_3.copy(), str(caminho_metadados or "")
