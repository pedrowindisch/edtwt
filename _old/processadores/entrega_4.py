from __future__ import annotations

import json
from typing import Callable

import numpy as np
import torch
from transformers import AutoModel, AutoTokenizer

from extrator.entrega import ENTREGA_4, ConfiguracaoEntrega, normalizar_texto_csv, para_json

COLUNAS_ENTREGA_4 = [
    "emb_bertimbau_input",
    "emb_bertimbau_mean",
]

MODEL_NAME = "neuralmind/bert-base-portuguese-cased"
BATCH_SIZE = 16
VECTOR_DIM = 768
MAX_LENGTH = 128


def _preparar_texto(linha: dict[str, str]) -> str:
    """Combina normalizacao_re com hashtags para formar o texto de entrada."""
    texto = normalizar_texto_csv(linha.get("normalizacao_re"))
    hashtags_raw = normalizar_texto_csv(linha.get("hashtags"))
    if hashtags_raw:
        hashtags = [h.strip() for h in hashtags_raw.split("|") if h.strip()]
        if hashtags:
            texto = (texto + " " + " ".join(hashtags)).strip()
    return texto


@torch.no_grad()
def _encode_mean(
    model,
    tokenizer,
    texts: list[str],
    batch_size: int,
    device: str,
) -> np.ndarray:
    """Gera embeddings com mean pooling usando BERTimbau."""
    all_vecs = []
    for i in range(0, len(texts), batch_size):
        batch = texts[i : i + batch_size]
        inputs = tokenizer(
            batch,
            padding=True,
            truncation=True,
            max_length=MAX_LENGTH,
            return_tensors="pt",
        ).to(device)
        outputs = model(**inputs)
        mask = inputs["attention_mask"].unsqueeze(-1).expand(outputs.last_hidden_state.size()).float()
        summed = torch.sum(outputs.last_hidden_state * mask, dim=1)
        counts = torch.clamp(mask.sum(dim=1), min=1e-9)
        vecs = (summed / counts).cpu().numpy()
        all_vecs.append(vecs)
    return np.vstack(all_vecs)


def processar_entrega_4(
    linhas: list[dict[str, str]],
    configuracao: ConfiguracaoEntrega,
    emitir_progresso: Callable[[int], None],
    emitir_status: Callable[[str], None],
) -> tuple[list[str], str]:
    emitir_status("Entrega 4: carregando BERTimbau...")

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModel.from_pretrained(MODEL_NAME)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)
    model.eval()

    total_linhas = len(linhas)

    # ── 1. Preparar textos de entrada ────────────────────────────────────────
    textos: list[str] = []
    for indice, linha in enumerate(linhas, start=1):
        if "normalizacao_re" not in linha:
            raise ValueError(
                "Entrega 4 precisa das colunas geradas pela Entrega 2, incluindo normalizacao_re."
            )
        texto = _preparar_texto(linha)
        linha["emb_bertimbau_input"] = texto
        textos.append(texto)
        emitir_progresso(int(indice / total_linhas * 15))

    # ── 2. Gerar embeddings com mean pooling ─────────────────────────────────
    emitir_status("Entrega 4: gerando embeddings BERTimbau (mean pooling)...")
    embeddings = _encode_mean(model, tokenizer, textos, BATCH_SIZE, device)
    emitir_progresso(70)

    # ── 3. Salvar embeddings no CSV ──────────────────────────────────────────
    emitir_status("Entrega 4: salvando embeddings...")
    for indice, (linha, vetor) in enumerate(zip(linhas, embeddings), start=1):
        linha["emb_bertimbau_mean"] = para_json(
            [round(float(v), 6) for v in vetor]
        )
        emitir_progresso(70 + int(indice / total_linhas * 25))

    # ── 4. Salvar metadados ──────────────────────────────────────────────────
    caminho_metadados = configuracao.caminho_metadados(ENTREGA_4)
    if caminho_metadados is not None:
        metadados = {
            "modelo": MODEL_NAME,
            "vector_size": VECTOR_DIM,
            "pooling": "mean",
            "batch_size": BATCH_SIZE,
            "max_length": MAX_LENGTH,
            "device": device,
        }
        caminho_metadados.write_text(
            json.dumps(metadados, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

    emitir_progresso(100)
    return COLUNAS_ENTREGA_4.copy(), str(caminho_metadados or "")
