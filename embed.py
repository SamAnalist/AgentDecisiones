"""
embed.py — Embeddings locales con RoBERTa‑BNE.
--------------------------------------------------------------------
• Genera vectores L2‑normalizados con Sentence‑Transformers.
• Procesa en lotes con un bucle simple; evita multi‑processing para compatibilidad
  con Windows (spawn) y entornos notebook.
• Si tu dataset es enorme y usas Linux, puedes re‑activar encode_multi_process()
  añadiendo `USE_MP=True` y ejecutando el script principal bajo
  `if __name__ == "__main__": …`.
"""

from typing import List

from sentence_transformers import SentenceTransformer
from langchain_core.embeddings import Embeddings

from config import EMBED_MODEL_ID

# ───────── Configurables ──────────────────────────────────────────
MAX_BATCH: int = 64   # tamaño del lote; 64 → ~1 GB RAM para 768‑dim

# ─────── Carga única del modelo ──────────────────────────────────
_model = SentenceTransformer(EMBED_MODEL_ID)

# ─────────────────────────────────────────────────────────────────
# Helpers internos
# ─────────────────────────────────────────────────────────────────

def _batchify(texts: List[str], batch: int):
    """Divide texts en lotes de tamaño *batch*."""
    for i in range(0, len(texts), batch):
        yield texts[i : i + batch]


def _encode(texts: List[str]):
    """Devuelve embeddings normalizados en batches secuenciales."""
    vectors = []
    for chunk in _batchify(texts, MAX_BATCH):
        vecs = _model.encode(
            chunk,
            normalize_embeddings=True,
            show_progress_bar=False,
        )
        vectors.extend(vecs)
    return [v.tolist() for v in vectors]


# ─────────────────────────────────────────────────────────────────
# LangChain adaptor
# ─────────────────────────────────────────────────────────────────

class BNEEmbeddings(Embeddings):
    """Adaptador LangChain simple sin multiproceso (compatible Windows)."""

    def embed_documents(self, texts: List[str]):
        return _encode(texts)

    def embed_query(self, text: str):
        return _model.encode(
            text,
            normalize_embeddings=True,
            show_progress_bar=False,
        ).tolist()
