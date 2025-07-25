import pyodbc
import pandas as pd
from datetime import datetime
import warnings

# Silence pandas warning
warnings.filterwarnings("ignore", category=UserWarning)

# Chunking parameters
CHUNK_SIZE = 200
CHUNK_OVERLAP = 50

def _split_text(text: str) -> list[str]:
    words = text.split()
    chunks = []
    i = 0
    while i < len(words):
        chunk = words[i : i + CHUNK_SIZE]
        chunks.append(" ".join(chunk))
        i += CHUNK_SIZE - CHUNK_OVERLAP
    return chunks

def fix_date(dt):
    """Ensure SQL Server compatible date or return None."""
    if isinstance(dt, pd.Timestamp):
        if dt.year < 1753:
            return None
        return dt.to_pydatetime()
    return None

# DB connection
conn = pyodbc.connect(
    "DRIVER={ODBC Driver 17 for SQL Server};"
    "SERVER=192.168.0.133;"
    "DATABASE=Reportes;"
    "Trusted_Connection=yes;"
)
cursor = conn.cursor()

# Load full documents
df = pd.read_sql("""
SELECT
    IdDocumento,
    TextoPDF,
    NUC,
    NumeroTramite,
    Sala,
    Tribunal,
    Materia,
    TipoFallo,
    TipoDocumento,
    FechaDecision,
    FechaTramite
FROM [Reportes].[IA].[JuritecaTrainingSample]
WHERE TextoPDF IS NOT NULL
""", conn)

# Load existing (IdDocumento, IdChunk) pairs
existing_chunks = pd.read_sql("""
    SELECT IdDocumento, IdChunk
    FROM [Reportes].[IA].[JuritecaChunks]
""", conn)
existing_pairs = set(zip(existing_chunks["IdDocumento"], existing_chunks["IdChunk"]))

print(f"📄 Total documentos para procesar: {len(df)}")

# Process document by document
for _, row in df.iterrows():
    doc_id = int(row["IdDocumento"])
    texto = row["TextoPDF"]

    if not isinstance(texto, str) or not texto.strip():
        continue

    try:
        chunks = _split_text(texto)
        print(f"🧩 ID {doc_id}: {len(chunks)} chunks")

        for idx, chunk in enumerate(chunks):
            if (doc_id, idx) in existing_pairs:
                continue  # Skip existing chunk

            try:
                cursor.execute("""
                    INSERT INTO [Reportes].[IA].[JuritecaChunks] (
                        IdDocumento, IdChunk, Texto,
                        NUC, NumeroTramite, Sala, Tribunal, Materia,
                        TipoFallo, TipoDocumento, FechaDecision, FechaTramite
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, (
                    doc_id, idx, chunk,
                    row["NUC"], row["NumeroTramite"], row["Sala"], row["Tribunal"],
                    row["Materia"], row["TipoFallo"], row["TipoDocumento"],
                    fix_date(row["FechaDecision"]), fix_date(row["FechaTramite"])
                ))
                conn.commit()
                print(f"✅ Inserted chunk {idx} for doc {doc_id}")

            except Exception as e_chunk:
                print(f"❌ Error inserting chunk {idx} of doc {doc_id}: {e_chunk}")
    except Exception as e_doc:
        print(f"❌ Error processing document {doc_id}: {e_doc}")

cursor.close()
conn.close()

print("✅ Chunking process completed with resume logic.")
