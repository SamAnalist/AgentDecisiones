# """
# build_index.py
# Genera dos índices FAISS:
#     • index_cases  –  chunks de sentencias (RoBERTa-BNE)
#     • index_laws   –  artículos de Constitución + Códigos (BART-Legal-base-es)
# """
#
# from pathlib import Path
# import pandas as pd
# from langchain_community.vectorstores import FAISS
#
# from config  import DATA_DIR, INDEX_DIR
# from chunker import make_chunks
# from embed   import BNEEmbeddings, get_embeddings   # get_embeddings("laws")
#
# # ───────── rutas fijas ────────────────────────────────────────────
# FILE_CASES   = DATA_DIR / "output (1).xlsx"
# INDEX_CASES  = INDEX_DIR / "index_cases"
#
# # ───────── índice de SENTENCIAS ──────────────────────────────────
# df_cases = pd.read_excel(FILE_CASES)
# df_cases["textoPDF"] = df_cases["textoPDF"].fillna("")
# docs_cases = make_chunks(df_cases)
#
# idx_cases = FAISS.from_documents(
#     docs_cases,
#     embedding=BNEEmbeddings(),          # RoBERTa-BNE
#     normalize_L2=True,
# )
# idx_cases.save_local(str(INDEX_CASES))
# print(f"✅ index_cases guardado ({len(docs_cases)} chunks)")
#

# build_index.py  — BLOQUE NUEVO para index_laws
# ------------------------------------------------
from pathlib import Path
# build_index.py  — BLOQUE CORREGIDO para index_laws
# --------------------------------------------------
from pathlib import Path
import pandas as pd
from langchain.schema import Document
from langchain_community.vectorstores import FAISS
from config import DATA_DIR, INDEX_DIR, CHUNK_SIZE, CHUNK_OVERLAP
from embed import get_embeddings

INDEX_LAWS = INDEX_DIR / "index_laws"
INDEX_LAWS.mkdir(parents=True, exist_ok=True)

# ---------- helpers ----------
def _split_text(text: str, size: int = CHUNK_SIZE, overlap: int = CHUNK_OVERLAP):
    words = str(text).split()
    i = 0
    while i < len(words):
        yield " ".join(words[i : i + size])
        i += size - overlap

def _chunks_from_records(records: list[dict]):
    docs = []
    for rec in records:
        raw = rec["text"]
        base = rec["metadata"]
        for idx, chunk in enumerate(_split_text(raw)):
            docs.append(Document(page_content=chunk,
                                 metadata={**base, "ChunkID": idx}))
    return docs

# ---------- 1. Constitución ----------
df_const = pd.read_csv(DATA_DIR / "constitucion.csv")
records_const = [
    {
        "text": row["ArticuloContenido"],
        "metadata": {"articulo": int(row["ArticuloNo"]),
                     "fuente": "constitucion"},
    }
    for _, row in df_const.iterrows()
    if pd.notna(row["ArticuloContenido"])
]

# ---------- 2. Criterios ----------
keep = [
    "ID","ItemID","Materia","Asunto","Categoria","SubCategoria","TipoDesicion",
    "Relevancia","RefRelevancia","BaseLegal","PalabrasClaves","NumDesicion",
    "FechaDoc","NumBoletin",
]
df_crit = pd.read_csv(DATA_DIR / "Criterios.csv", usecols=keep + ["Título","Resena"])
records_crit = []
for _, row in df_crit.iterrows():
    if pd.isna(row["Título"]) and pd.isna(row["Resena"]):
        continue
    texto = f"{row['Título'] or ''}. {row['Resena'] or ''}"
    meta  = {k: row[k] for k in keep}
    meta["fuente"] = "criterio"
    records_crit.append({"text": texto, "metadata": meta})

# ---------- 3. Chunkear y embebir ----------
docs = _chunks_from_records(records_const + records_crit)

idx_laws = FAISS.from_documents(
    docs,
    embedding=get_embeddings("laws"),
    normalize_L2=True,
)
idx_laws.save_local(str(INDEX_LAWS))
print(f"✅ index_laws guardado ({len(docs)} chunks)")
