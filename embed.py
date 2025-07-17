# embed.py

from sentence_transformers import SentenceTransformer
from config import EMBED_MODEL_ID
from typing import List
from langchain_huggingface import HuggingFaceEmbeddings
_model = SentenceTransformer(EMBED_MODEL_ID)   # ← ESTA línea debe existir


# Tu wrapper para RoBERTa-BNE sigue igual:
from langchain_core.embeddings import Embeddings

# ---------------------------

def get_embeddings(kind: str = "cases"):
    if kind == "laws":
        # Usamos el wrapper oficial con normalización L2
        return HuggingFaceEmbeddings(
            model_name="mrm8488/bart-legal-base-es",
            model_kwargs={"device": "cpu"},              # o "cuda"
            encode_kwargs={"normalize_embeddings": True},# ← muy importante
        )
    # Si es "cases", seguimos con tu wrapper local
    return BNEEmbeddings()


# ─────────────────────────────────────────────────────────────────
# Helpers internos
# ─────────────────────────────────────────────────────────────────

def _batchify(texts: List[str], batch: int):
    """Divide texts en lotes de tamaño *batch*."""
    for i in range(0, len(texts), batch):
        yield texts[i : i + batch]


def _encode(texts: List[str], MAX_BATCH=None):
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
