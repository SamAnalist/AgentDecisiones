"""
Script: excel_resume_text_only.py

Qué hace:
- Lee un Excel base (EXCEL_PATH) con columna 'UrlDocumentoFirmado'.
- Si existe <base>_procesado.xlsx, reanuda desde ahí.
- Para filas con textoPDF vacío:
    * Descarga el PDF
    * Extrae SOLO la primera página:
        - Capa de texto (PyMuPDF)
        - Si no hay texto y tienes Tesseract configurado → OCR 300dpi
    * Limpia el texto para Excel y lo guarda en 'textoPDF'
- Ignora filas que fallen o que queden en blanco.
- Guarda progreso cada N filas (SAVE_EVERY).

Requisitos:
    pandas openpyxl requests pymupdf pillow (opcional: pytesseract)
"""

from __future__ import annotations
import os, io, re, pathlib, sys
import pandas as pd
import requests
import fitz  # PyMuPDF
from PIL import Image

# ========== CONFIG ==========
EXCEL_PATH = r"C:\Users\samfernandez\Downloads\DecisionesConFechasErroneas.xlsx"
COL_URL    = "UrlDocumentoFirmado"
SAVE_EVERY = 1   # guarda cada N filas

# OCR opcional (si tienes Tesseract; si no, déjalo así)
_USE_OCR = True
try:
    import pytesseract
    _TESSERACT_CMD = r"C:\Users\su-samfernandez\PycharmProjects\ExtractorDeInvolucradosPJ\tesseract.exe"
    if _TESSERACT_CMD:
        pytesseract.pytesseract.tesseract_cmd = _TESSERACT_CMD
        os.environ["TESSDATA_PREFIX"] = os.path.join(os.path.dirname(_TESSERACT_CMD), "tessdata")
    _TESS_LANG = os.getenv("TESSERACT_LANG", "spa+eng")
except Exception:
    _USE_OCR = False
# ===========================

# ---- Limpieza para Excel (evita IllegalCharacterError) ----
try:
    from openpyxl.cell.cell import ILLEGAL_CHARACTERS_RE
except Exception:
    ILLEGAL_CHARACTERS_RE = re.compile(r"[\x00-\x08\x0B-\x0C\x0E-\x1F]")
EXCEL_CELL_MAX = 32767

def clean_for_excel(s: str) -> str:
    if s is None:
        return ""
    s = s.replace("\xa0", " ").replace("\x0c", "\n")
    s = ILLEGAL_CHARACTERS_RE.sub("", s)
    if len(s) > EXCEL_CELL_MAX - 1:
        s = s[:EXCEL_CELL_MAX - 1]
    return s
# -----------------------------------------------------------

def extract_first_page_text(pdf_bytes: bytes) -> str:
    """
    Extrae SOLO la primera página:
      - primero capa de texto
      - si vacío y OCR habilitado: Tesseract 300dpi
    Devuelve texto ya saneado para Excel.
    """
    text = ""
    doc = fitz.open(stream=pdf_bytes, filetype="pdf")
    try:
        if doc.page_count == 0:
            return ""
        page = doc.load_page(0)
        text = page.get_text("text").strip()
        if not text and _USE_OCR:
            pix = page.get_pixmap(dpi=300)
            img = Image.open(io.BytesIO(pix.tobytes("png")))
            import pytesseract  # seguro de estar importado
            text = pytesseract.image_to_string(img, lang=_TESS_LANG).strip()
    finally:
        doc.close()
    return clean_for_excel(text or "")

def _suffix_path(path: pathlib.Path, suffix: str, new_ext: str = "xlsx") -> pathlib.Path:
    return path.with_name(f"{path.stem}{suffix}.{new_ext}")

def _load_for_resume(in_path: pathlib.Path) -> tuple[pd.DataFrame, pathlib.Path]:
    out_path = _suffix_path(in_path, "_procesado", "xlsx")
    if out_path.exists():
        print(f"[Reanudar] Cargando existente: {out_path}")
        df = pd.read_excel(out_path)
    else:
        df = pd.read_excel(in_path)

    if "textoPDF" not in df.columns:
        df["textoPDF"] = ""

    return df, out_path

def _safe_save(df: pd.DataFrame, out_path: pathlib.Path) -> None:
    # limpieza general por si acaso
    for col in df.select_dtypes(include=["object"]).columns:
        df[col] = df[col].map(lambda v: clean_for_excel(v) if isinstance(v, str) else v)
    with pd.ExcelWriter(out_path, engine="openpyxl") as writer:
        df.to_excel(writer, index=False)

def process_excel_text_only(excel_path: str) -> pathlib.Path:
    in_path = pathlib.Path(excel_path)
    if not in_path.exists():
        raise FileNotFoundError(f"No existe: {in_path}")

    df, out_path = _load_for_resume(in_path)

    if COL_URL not in df.columns:
        raise ValueError(f"Falta columna requerida: '{COL_URL}'")

    session = requests.Session()
    session.headers.update({"User-Agent": "DocFirstPageExtractor/1.0"})
    timeout = (6, 25)

    processed_since_save = 0
    total = len(df)

    for idx, row in df.iterrows():
        url = str(row.get(COL_URL) or "").strip()
        texto_actual = str(row.get("textoPDF") or "")

        # Reanudar: solo procesa si está vacío
        if texto_actual.strip():
            continue

        if not url:
            # sin URL → ignora
            continue

        print(f"[{idx+1}/{total}] {url[:100]}{'...' if len(url)>100 else ''}")
        try:
            r = session.get(url, timeout=timeout)
            r.raise_for_status()
            pdf_bytes = r.content
            first_text = extract_first_page_text(pdf_bytes)
            # si quedó vacío, ignorar (no escribir nada)
            if first_text.strip():
                df.at[idx, "textoPDF"] = first_text
        except Exception as e:
            # ignora y sigue
            print(f"   - [IGNORADO] {type(e).__name__}: {e}")

        processed_since_save += 1
        if processed_since_save >= SAVE_EVERY:
            try:
                _safe_save(df, out_path)
                processed_since_save = 0
                print(f"   ✓ Progreso guardado en: {out_path.name}")
            except Exception as e:
                print(f"   [WARN] Fallo guardando (se continúa): {type(e).__name__}: {e}")

    _safe_save(df, out_path)
    print(f"[FIN] Archivo listo: {out_path}")
    return out_path

if __name__ == "__main__":
    path = EXCEL_PATH if len(sys.argv) == 1 else sys.argv[1]
    try:
        out = process_excel_text_only(path)
        print(f"Listo: {out}")
    except Exception as e:
        print(f"ERROR: {type(e).__name__}: {e}")
        sys.exit(1)
