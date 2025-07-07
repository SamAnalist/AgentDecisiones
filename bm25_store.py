"""
bm25_store.py
-------------
Índice BM25 ligero en memoria para búsqueda léxica.

Se basa en Whoosh (puro-Python, sin dependencias de system-libraries),
ideal mientras sigas trabajando sólo con archivos Excel locales.

• build_bm25()   → crea o carga un índice Whoosh en DATA_DIR / bm25.idx
• search_bm25(q) → devuelve lista de (DocumentID, score) top-k

Si luego migras a Elasticsearch / OpenSearch bastará con reemplazar
estas dos funciones sin tocar el resto del código.
"""

import os
from pathlib import Path
from typing import List, Tuple

import pandas as pd
from whoosh import index
from whoosh.fields import ID, TEXT, Schema
from whoosh.qparser import MultifieldParser

from config import DATA_DIR

BM25_DIR = DATA_DIR / "bm25.idx"
SCHEMA = Schema(
    DocumentID=ID(stored=True, unique=True),
    texto=TEXT(stored=False, phrase=True),
)


def build_bm25(df: pd.DataFrame) -> None:
    """Construye el índice BM25 o lo actualiza si ya existe."""
    if not BM25_DIR.exists():
        os.makedirs(BM25_DIR, exist_ok=True)
        ix = index.create_in(BM25_DIR, SCHEMA)
    else:
        ix = index.open_dir(BM25_DIR)

    writer = ix.writer()
    for _, row in df.iterrows():
        writer.update_document(
            DocumentID=str(int(row["IdDocumento"])),
            texto=row["textoPDF"][:10_000],  # límite razonable
        )
    writer.commit()
    print("✓ BM25 index listo en", BM25_DIR)


def search_bm25(
    query: str, k: int = 10
) -> List[Tuple[int, float]]:
    """Busca en el índice Whoosh y devuelve (DocumentID, score)."""
    if not BM25_DIR.exists():
        raise RuntimeError("BM25 index no construido. Ejecuta build_bm25().")

    ix = index.open_dir(BM25_DIR)
    with ix.searcher() as s:
        parser = MultifieldParser(["texto"], schema=SCHEMA)
        q = parser.parse(query)
        results = s.search(q, limit=k)
        return [(int(r["DocumentID"]), r.score) for r in results]
