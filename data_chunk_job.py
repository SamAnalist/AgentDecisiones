import pyodbc
import pandas as pd

# Parámetros para chunking
CHUNK_SIZE = 500
CHUNK_OVERLAP = 50

def _split_text(text: str) -> list[str]:
    """Divide texto en chunks con solapamiento."""
    words = text.split()
    chunks = []
    i = 0
    while i < len(words):
        chunk = words[i : i + CHUNK_SIZE]
        chunks.append(" ".join(chunk))
        i += CHUNK_SIZE - CHUNK_OVERLAP
    return chunks

# Conexión a la base de datos
conn = pyodbc.connect(
    "DRIVER={ODBC Driver 17 for SQL Server};"
    "SERVER=192.168.0.133;"
    "DATABASE=Reportes;"
    "Trusted_Connection=yes;"
)
cursor = conn.cursor()

# Leer documentos procesados (con texto extraído)
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

# Procesar cada documento
for _, row in df.iterrows():
    texto = row["TextoPDF"]
    if not isinstance(texto, str) or not texto.strip():
        continue

    # Generar los chunks
    chunks = _split_text(texto)

    for idx, chunk in enumerate(chunks):
        cursor.execute("""
            INSERT INTO [Reportes].[IA].[JuritecaChunks] (
                IdDocumento, IdChunk, Texto,
                NUC, NumeroTramite, Sala, Tribunal, Materia,
                TipoFallo, TipoDocumento, FechaDecision, FechaTramite
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            int(row["IdDocumento"]), idx, chunk,
            row["NUC"], row["NumeroTramite"], row["Sala"], row["Tribunal"],
            row["Materia"], row["TipoFallo"], row["TipoDocumento"],
            row["FechaDecision"], row["FechaTramite"]
        ))
        conn.commit()

print("✅ Chunks insertados correctamente.")

cursor.close()
conn.close()
