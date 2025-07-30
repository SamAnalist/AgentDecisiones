"""
chunker.py  ─  genera chunks con metadatos completos (+ fechas)

• Lee el Excel `output.xlsx` desde DATA_DIR.
• Divide la columna `textoPDF` en trozos de longitud CHUNK_SIZE tokens
  con solapamiento CHUNK_OVERLAP.
• Guarda cada chunk como JSON individual en DATA_DIR/chunks/.
• Metadatos incluidos por chunk:
    - DocumentID
    - NUC
    - NumeroTramite
    - Sala
    - Tribunal
    - Materia
    - TipoFallo
    - TipoDocumento
    - FechaDecision     ←  NEW
    - FechaTramite      ←  NEW
    - ChunkID
"""

import os
import json
import pandas as pd
from langchain.schema import Document
from config import DATA_DIR, CHUNK_SIZE, CHUNK_OVERLAP
from tqdm import tqdm  # Asegúrate de tener esto instalado: pip install tqdm

CHUNKS_DIR = DATA_DIR / "chunks"
os.makedirs(CHUNKS_DIR, exist_ok=True)

def _split_text(text: str) -> list[str]:
    """Divide el texto en trozos de tamaño fijo con solapamiento."""
    words = text.split()
    chunks = []
    i = 0
    while i < len(words):
        chunk = words[i : i + CHUNK_SIZE]
        chunks.append(" ".join(chunk))
        i += CHUNK_SIZE - CHUNK_OVERLAP
    return chunks

def make_chunks(df: pd.DataFrame) -> list[Document]:
    docs: list[Document] = []
    print(f"📄 Procesando {len(df)} filas del DataFrame...")

    for i, row in tqdm(df.iterrows(), total=len(df), desc="🔨 Chunking"):
        raw_text = row.get("textoPDF", "")
        if not isinstance(raw_text, str) or not raw_text.strip():
            print(f"⚠️  Fila {i} sin texto, se omite.")
            continue

        try:
            fecha_dec = (
                str(row["FechaDecision"].date()) if pd.notna(row.get("FechaDecision")) else ""
            )
            fecha_tra = (
                str(row["FechaTramite"].date()) if pd.notna(row.get("FechaTramite")) else ""
            )

            chunks = _split_text(raw_text)
            print(f"🧩 Documento {row['IdDocumento']} → {len(chunks)} chunks")

            for idx, chunk in enumerate(chunks):
                meta = {
                    "DocumentID": int(row["IdDocumento"]),
                    "NUC": row.get("NUC", ""),
                    "NumeroTramite": row.get("NumeroTramite", ""),
                    "Sala": row.get("Sala", ""),
                    "Tribunal": row.get("Tribunal", ""),
                    "Materia": row.get("Materia", ""),
                    "TipoFallo": row.get("TipoFallo", ""),
                    "TipoDocumento": row.get("TipoDocumento", ""),
                    "FechaDecision": fecha_dec,
                    "FechaTramite": fecha_tra,
                    "ChunkID": idx,
                }

                docs.append(Document(page_content=chunk, metadata=meta))

                out_path = os.path.join(
                    CHUNKS_DIR, f"{meta['DocumentID']}_chunk_{idx}.json"
                )
                with open(out_path, "w", encoding="utf-8") as f:
                    json.dump({"text": chunk, "metadata": meta}, f, ensure_ascii=False)

        except Exception as e:
            print(f"❌ Error procesando fila {i}: {e}")

    print(f"✅ Proceso finalizado. Total de chunks generados: {len(docs)}")
    return docs

if __name__ == "__main__":
    excel_path = os.path.join(DATA_DIR, "output (1).xlsx")
    print(f"📥 Cargando archivo: {excel_path}")
    df = pd.read_excel(excel_path)
    total_docs = make_chunks(df)
    print(f"✅ Chunks generados: {len(total_docs)}. Guardados en {CHUNKS_DIR}")
