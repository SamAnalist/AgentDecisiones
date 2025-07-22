"""
build_index.py
Genera dos índices FAISS:
    • index_cases  –  chunks de sentencias (RoBERTa-BNE)
    • index_laws   –  artículos de Constitución + Códigos (BART-Legal-base-es)
"""

from pathlib import Path
import pandas as pd
from langchain_community.vectorstores import FAISS

from config  import DATA_DIR, INDEX_DIR
from chunker import make_chunks
from embed   import BNEEmbeddings, get_embeddings   # get_embeddings("laws")

# ───────── rutas fijas ────────────────────────────────────────────
FILE_CASES   = DATA_DIR / "output (1).xlsx"
FILE_CODIGO  = DATA_DIR / "codigo.csv"
FILE_CONST   = DATA_DIR / "constitucion.csv"

INDEX_CASES  = INDEX_DIR / "index_cases"
INDEX_LAWS   = INDEX_DIR / "index_laws"

# ───────── índice de SENTENCIAS ──────────────────────────────────
df_cases = pd.read_excel(FILE_CASES)
df_cases["textoPDF"] = df_cases["textoPDF"].fillna("")
docs_cases = make_chunks(df_cases)

idx_cases = FAISS.from_documents(
    docs_cases,
    embedding=BNEEmbeddings(),          # RoBERTa-BNE
    normalize_L2=True,
)
idx_cases.save_local(str(INDEX_CASES))
print(f"✅ index_cases guardado ({len(docs_cases)} chunks)")
