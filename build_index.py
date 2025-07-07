"""
build_index.py — Genera el índice FAISS con embeddings RoBERTa-BNE.
"""

import pandas as pd
from langchain_community.vectorstores import FAISS
from config import DATA_DIR, INDEX_DIR
from chunker import make_chunks
from embed import BNEEmbeddings     # ⬅️  nuevo wrapper

# 1. Leer Excel y limpiar NaN
df = pd.read_excel(DATA_DIR / "output (1).xlsx")
df["textoPDF"] = df["textoPDF"].fillna("")

# 2. Generar documentos chunk-level
docs = make_chunks(df)

# 3. Embeddings
emb = BNEEmbeddings()

# 4. Construir índice (FlatL2 + normalize → coseno)
vectordb = FAISS.from_documents(
    docs,
    embedding=emb,
    normalize_L2=True,
)

vectordb.save_local(str(INDEX_DIR))
print("✅ Índice FAISS guardado en", INDEX_DIR)
