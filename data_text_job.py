# chunker_to_sql.py

import os
import pyodbc
import pandas as pd
import requests
import fitz  # PyMuPDF
from PIL import Image
import pytesseract
import io

# Configurar ruta de Tesseract manualmente
TESSERACT_CMD = (
    r"C:\Users\su-samfernandez\PycharmProjects\ExtractorDeInvolucradosPJ\tesseract.exe"
)
os.environ["TESSERACT_CMD"] = TESSERACT_CMD
os.environ["TESSDATA_PREFIX"] = os.path.join(os.path.dirname(TESSERACT_CMD), "tessdata")
pytesseract.pytesseract.tesseract_cmd = TESSERACT_CMD  # línea clave

# Configuración de conexión
conn = pyodbc.connect(
    "DRIVER={ODBC Driver 17 for SQL Server};"
    "SERVER=192.168.0.133;"
    "DATABASE=Reportes;"
    "Trusted_Connection=yes;"
)
cursor = conn.cursor()

# Leer los textos ya procesados
df = pd.read_sql("""
SELECT IdDocumento, UrlDocumentoFirmado
FROM [Reportes].[IA].[JuritecaTrainingSample]
WHERE TextoPDF IS NULL AND UrlDocumentoFirmado IS NOT NULL
""", conn)

def extract_text_from_pdf_bytes(pdf_bytes: bytes) -> str:
    doc = fitz.open(stream=pdf_bytes, filetype="pdf")
    all_text = []
    for page in doc:
        text = page.get_text("text")
        if text.strip():
            all_text.append(text)
        else:
            pix = page.get_pixmap(dpi=300)
            img = Image.open(io.BytesIO(pix.tobytes("png")))
            ocr_text = pytesseract.image_to_string(img)
            all_text.append(ocr_text)
    doc.close()
    return "\n".join(all_text).strip()

# Procesar fila por fila
for _, row in df.iterrows():
    doc_id = row["IdDocumento"]
    url = row["UrlDocumentoFirmado"]

    try:
        print(f"Procesando ID {doc_id}...")
        response = requests.get(url, timeout=20)
        response.raise_for_status()

        texto = extract_text_from_pdf_bytes(response.content)

        cursor.execute("""
            UPDATE [Reportes].[IA].[JuritecaTrainingSample]
            SET TextoPDF = ?
            WHERE IdDocumento = ?
        """, texto, doc_id)
        conn.commit()
        print(f"✅ Texto guardado para ID {doc_id}")

    except Exception as e:
        print(f"❌ Error con ID {doc_id}: {e}")

cursor.close()
conn.close()
