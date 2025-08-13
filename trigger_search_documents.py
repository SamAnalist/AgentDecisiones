
import os
import pyodbc
import pandas as pd
import requests
import fitz  # PyMuPDF
from PIL import Image
import pytesseract
import io
import io, pymupdf, pymupdf4llm
from PIL import Image
import pytesseract

def extract_markdown(pdf_bytes: bytes, ocr_fallback=True) -> str:
    doc = pymupdf.open(stream=pdf_bytes, filetype="pdf")
    # ⇣ extrae a Markdown; page_chunks=True devuelve lista/dicts si prefieres
    md = pymupdf4llm.to_markdown(
        doc,
        page_chunks=False,      # True ⇒ dicta por página
        ignore_images=True,     # no insertes imágenes como base64
        table_strategy="lines", # detecta tablas clásicas
        show_progress=False
    )  # ⇒ str con \n\n### Encabezados, listas, tablas ...
    doc.close()

    # Fallback: si no encontró texto y quieres OCR
    if ocr_fallback and not md.strip():
        ocr_pages = []
        doc = pymupdf.open(stream=pdf_bytes, filetype="pdf")
        for page in doc:
            pix = page.get_pixmap(dpi=300)
            img = Image.open(io.BytesIO(pix.tobytes("png")))
            ocr_pages.append(pytesseract.image_to_string(img))
        doc.close()
        md = "\n".join(ocr_pages)

    return md

# Configurar ruta de Tesseract manualmente
TESSERACT_CMD = (
    r"C:\Users\su-samfernandez\PycharmProjects\ExtractorDeInvolucradosPJ\tesseract.exe"
)
os.environ["TESSERACT_CMD"] = TESSERACT_CMD
os.environ["TESSDATA_PREFIX"] = os.path.join(os.path.dirname(TESSERACT_CMD), "tessdata")
pytesseract.pytesseract.tesseract_cmd = TESSERACT_CMD  # línea clave

# Configuración de conexión
conn = pyodbc.connect(
    "DRIVER={ODBC Driver 18 for SQL Server};"
    "SERVER=192.168.0.133;"
    "DATABASE=DepositoDocumentos;"
    "UID=su-samfernandez;"
    "PWD=Temporal100*;"
    "Encrypt=yes;"
    "TrustServerCertificate=yes;"
    # opcional: "Timeout=5;"
)

cursor = conn.cursor()
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

def colectar_texto(nuc):
    # Leer los textos ya procesados
    df = pd.read_sql(f"""
    SELECT c.NUC, t.NumeroTramite, d.URL, d.FechaCreacion
FROM DepositoDocumentos.SJ.casos c 
JOIN DepositoDocumentos.SJ.Tramites t ON t.IdCaso = c.IdCaso AND t.activo = 1
JOIN DepositoDocumentos.SJ.Documentos d ON d.IdTramite = t.IdTramite AND d.activo = 1
WHERE c.activo = 1 AND c.NUC = '{nuc}'
ORDER BY d.FechaCreacion
    """, conn)
    # Procesar fila por fila
    # Procesar fila por fila (robusto: siempre escribe una celda)
    df["texto_pdf"] = ""  # pre-crea la columna
    for idx, row in df.iterrows():
        doc_id = row["NUC"]
        url = row["URL"]
        texto = ""
        try:
            print(f"Procesando ID {doc_id}…")
            response = requests.get(url, timeout=(6, 25))  # (connect, read)
            response.raise_for_status()
            texto = extract_markdown(response.content) or ""
            print(f"✅ Texto guardado para ID {doc_id} (len={len(texto)})")
        except Exception as e:
            print(f"❌ Error con ID {doc_id}: {e}")  # deja texto=""
        df.at[idx, "texto_pdf"] = texto

    return df

print(colectar_texto(nuc="034-2021-ECON-00366"))