"""
build_index.py
Crea / reemplaza dos índices FAISS:
    • index_cases  –  chunks JSON de sentencias  (RoBERTa-BNE)
    • index_laws   –  Constitución + Códigos,     chunk ≈750 tokens, BART-Legal
"""

from pathlib import Path
import json, os
import pandas as pd
from langchain_community.vectorstores import FAISS
from langchain.schema import Document

from config  import DATA_DIR, INDEX_DIR, CHUNK_SIZE, CHUNK_OVERLAP
from embed   import BNEEmbeddings, get_embeddings                # get_embeddings("laws")
from langchain.text_splitter import RecursiveCharacterTextSplitter

# ───────────── rutas fijas ────────────────────────────────────────
CHUNKS_DIR  = DATA_DIR / "chunks"              # ya creado por chunker.py
FILE_CODIGO = DATA_DIR / "codigo.csv"
FILE_CONST  = DATA_DIR / "constitucion.csv"

INDEX_CASES = INDEX_DIR / "index_cases"
INDEX_LAWS  = INDEX_DIR / "index_laws"

# ───────────── índice CASES  (usa chunks JSON existentes) ─────────
# docs_cases = []
# for fn in CHUNKS_DIR.glob("*.json"):
#     with open(fn, encoding="utf-8") as f:
#         j = json.load(f)
#     docs_cases.append(
#         Document(page_content=j["text"], metadata=j.get("metadata", {}))
#     )
#
# idx_cases = FAISS.from_documents(docs_cases, BNEEmbeddings(), normalize_L2=True)
# idx_cases.save_local(str(INDEX_CASES))
# print(f"✅ index_cases guardado ({len(docs_cases)} chunks)")

# ───────────── helper para trocear artículos de leyes ─────────────
splitter = RecursiveCharacterTextSplitter(
    chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP
)

def _csv_to_chunks(path: Path, fuente: str):
    df = pd.read_csv(path)
    col_texto = "ArticuloContenido" if "ArticuloContenido" in df.columns else "Articulo"
    col_num   = "ArticuloNo"         if "ArticuloNo" in df.columns else "Noarticulo"
    chunks = []
    for _, row in df.iterrows():
        art_num  = str(row[col_num]) if col_num in df else ""
        art_text = str(row[col_texto])
        for chunk in splitter.split_text(art_text):
            chunks.append(
                Document(
                    page_content=chunk,
                    metadata={"articulo": f"Art. {art_num}", "fuente": fuente}
                )
            )
    return chunks

# ───────────── índice LAWS (Constitución + Códigos) ───────────────
docs_laws = _csv_to_chunks(FILE_CODIGO, "codigo.csv") + \
            _csv_to_chunks(FILE_CONST,  "constitucion.csv")

idx_laws = FAISS.from_documents(
    docs_laws,
    embedding=get_embeddings("laws"),      # BART-Legal
    normalize_L2=True,
)
idx_laws.save_local(str(INDEX_LAWS))
print(f"✅ index_laws guardado ({len(docs_laws)} sub-chunks)")
