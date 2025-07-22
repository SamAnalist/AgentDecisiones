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
            model_name=EMBED_MODEL_ID,
            model_kwargs={"device": "cpu"},              # o "cuda"
            encode_kwargs={"normalize_embeddings": True},# ← muy importante
        )
    # Si es "cases", seguimos con tu wrapper local
    return BNEEmbeddings()


# ─────────────────────────────────────────────────────────────────
# Helpers internos
# ─────────────────────────────────────────────────────────────────
# embed.py  (ajusta solo estas partes)

MAX_BATCH_SIZE = 64   # ←  número seguro para CPU y Windows

def _batchify(texts: List[str], batch: int | None):
    """Divide texts en lotes de tamaño *batch* (o todo si batch es None)."""
    step = batch or len(texts)          # usa lista entera si batch==None
    for i in range(0, len(texts), step):
        yield texts[i : i + step]

def _encode(texts: List[str], MAX_BATCH: int | None = None):
    """Devuelve embeddings normalizados por lotes."""
    batch_size = MAX_BATCH or MAX_BATCH_SIZE
    vectors = []
    for chunk in _batchify(texts, batch_size):
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
