"""
vectorstore.py — Carga el índice FAISS y expone helpers de búsqueda.
"""

from langchain_community.vectorstores import FAISS
from config import INDEX_DIR
from embed import BNEEmbeddings      # ⬅️  nuevo wrapper

emb = BNEEmbeddings()
# Ruta correcta al sub-directorio que contiene index.faiss / index.pkl
INDEX_CASES_DIR = INDEX_DIR / "index_cases"   # ✅

vectordb = FAISS.load_local(
    str(INDEX_CASES_DIR),                     # ← aquí el path correcto
    embeddings=emb,
    allow_dangerous_deserialization=True,
)


# ✏️ EDITA la cabecera
from langchain_community.vectorstores import FAISS
from config import INDEX_DIR
from embed import BNEEmbeddings, get_embeddings       # ➕ NUEVO


# # ➕ NUEVO  índice de leyes
# try:
#     lawdb = FAISS.load_local(
#         str(INDEX_DIR/"index_laws"),
#         embeddings=get_embeddings("laws"),
#         allow_dangerous_deserialization=True,
#     )
# except FileNotFoundError:
#     lawdb = None   # todavía no generado


# ───── Helpers ───────────────────────────────────────────────────────
def search_by_text(text: str, k: int = 5, filtro: dict | None = None):
    """Búsqueda con filtro opcional por metadatos (e.g. {'Materia':'Penal'})."""
    return vectordb.similarity_search(text, k=k, filter=filtro)

def search_by_vector(vec, k: int = 5, filtro: dict | None = None):
    return vectordb.similarity_search_by_vector(vec, k=k, filter=filtro)


# def law_search(text: str, k: int = 5):
#     if not lawdb:
#         return []
#     return lawdb.similarity_search(text, k=k)
lawdb = FAISS.load_local(
    str(INDEX_DIR / "index_laws"),
    embeddings=get_embeddings("laws"),
    allow_dangerous_deserialization=True,
)

def law_search(text: str, k: int = 5, filtro: dict | None = None):
    return lawdb.similarity_search(text, k=k, filter=filtro)