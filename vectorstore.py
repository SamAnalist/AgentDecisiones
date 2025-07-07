"""
vectorstore.py — Carga el índice FAISS y expone helpers de búsqueda.
"""

from langchain_community.vectorstores import FAISS
from config import INDEX_DIR
from embed import BNEEmbeddings      # ⬅️  nuevo wrapper

emb = BNEEmbeddings()

vectordb = FAISS.load_local(
    str(INDEX_DIR),
    embeddings=emb,
    allow_dangerous_deserialization=True,  # usa pickle para metadatos
)

# ───── Helpers ───────────────────────────────────────────────────────
def search_by_text(text: str, k: int = 5, filtro: dict | None = None):
    """Búsqueda con filtro opcional por metadatos (e.g. {'Materia':'Penal'})."""
    return vectordb.similarity_search(text, k=k, filter=filtro)

def search_by_vector(vec, k: int = 5, filtro: dict | None = None):
    return vectordb.similarity_search_by_vector(vec, k=k, filter=filtro)
