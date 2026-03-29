from __future__ import annotations

import json
from typing import Callable

from sklearn.feature_extraction.text import TfidfVectorizer

from extrator.entrega import ENTREGA_2, ConfiguracaoEntrega, normalizar_com_regex, normalizar_texto_csv, para_json


COLUNAS_ENTREGA_2 = [
    "normalizacao_re",
    "features_tfidf_sklearn",
]


def processar_entrega_2(
    linhas: list[dict[str, str]],
    configuracao: ConfiguracaoEntrega,
    emitir_progresso: Callable[[int], None],
    emitir_status: Callable[[str], None],
) -> tuple[list[str], str]:
    emitir_status(
        "Entrega 2: normalizacao com re e selecao de features com scikit-learn..."
    )

    total_linhas = len(linhas)
    textos_normalizados: list[str] = []
    vetorizador = TfidfVectorizer(
        lowercase=False,
        max_df=0.85,
        min_df=2,
        max_features=1000,
        token_pattern=r"(?u)\b\w\w+\b",
    )

    for indice, linha in enumerate(linhas, start=1):
        if "stemming_nltk" not in linha:
            raise ValueError(
                "Entrega 2 precisa das colunas geradas pela Entrega 1, incluindo stemming_nltk."
            )

        texto_base = normalizar_texto_csv(linha.get("stemming_nltk"))
        texto_normalizado = normalizar_com_regex(texto_base)

        linha["normalizacao_re"] = texto_normalizado
        textos_normalizados.append(texto_normalizado)
        emitir_progresso(int(indice / total_linhas * 50))

    nomes_features: list[str] = []
    features_por_linha: list[str] = [para_json({}) for _ in linhas]

    if any(textos_normalizados):
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
                emitir_progresso(50 + int(indice / total_linhas * 50))
        except ValueError:
            emitir_progresso(100)
    else:
        emitir_progresso(100)

    for linha, mapa_features in zip(linhas, features_por_linha):
        linha["features_tfidf_sklearn"] = mapa_features

    caminho_metadados = configuracao.caminho_metadados(ENTREGA_2)
    if caminho_metadados is not None:
        params = vetorizador.get_params()
        caminho_metadados.write_text(
            json.dumps(
                {
                    "feature_names": nomes_features,
                    "vectorizer": {
                        "max_df": params["max_df"],
                        "min_df": params["min_df"],
                        "max_features": params["max_features"],
                        "token_pattern": params["token_pattern"],
                        "lowercase": params["lowercase"],
                    },
                },
                ensure_ascii=False,
                indent=2,
            ),
            encoding="utf-8",
        )

    return COLUNAS_ENTREGA_2.copy(), str(caminho_metadados or "")
