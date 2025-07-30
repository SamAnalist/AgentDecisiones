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

print("🔧 Probando modelo de embeddings...")
model = BNEEmbeddings()
print("✅ Modelo de embeddings cargado")

if not FILE_CASES.exists():
    print(f"❌ Archivo no encontrado: {FILE_CASES}")

try:
    print("🔧 Cargando archivo Excel...")
    df_cases = pd.read_excel(FILE_CASES)
    print("🔧 Generando chunks...")    
    df_cases["textoPDF"] = df_cases["textoPDF"].fillna("")
    docs_cases = make_chunks(df_cases)
    print("G✅ chunks generados ({len(docs_cases)} chunks)")
    idx_cases = FAISS.from_documents(
        docs_cases,
        embedding=model,          # RoBERTa-BNE
        normalize_L2=True,
    )
    idx_cases.save_local(str(INDEX_CASES))
    print(f"✅ index_cases guardado ({len(docs_cases)} chunks)")
except Exception as e:
    print(f"❌ index_cases no guardado: {e}")
    pass

# ───────── índice de SENTENCIAS ──────────────────────────────────

