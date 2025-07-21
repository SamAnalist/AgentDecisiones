"""
tools/expediente.py
-------------------
Herramienta “expediente”: busca un NUC o IdDocumento,
recupera los chunks relacionados y genera un JSON con:
  - resumen
  - considerandos
  - fallo_literal
"""

import re
import json
import pandas as pd
from together import Together
from config import DATA_DIR, TOGETHER_API_KEY, LLM_MODEL_ID
from embed import BNEEmbeddings
from vectorstore import search_by_vector
emb = BNEEmbeddings()
# ── 1. Cargar y normalizar DataFrame ───────────────────────────────
_df = pd.read_excel(DATA_DIR / "output (1).xlsx")

# Normalizar nombres: quitar espacios y mantener mayúsculas/minúsculas exactas
_df.columns = [c.strip() for c in _df.columns]

# Alias posibles → nombre estándar
_ALIAS = {
    "nuc": "NUC",
    "numero de caso": "NUC",
    "numeroexpediente": "NUC",
    "iddocumento": "IdDocumento",
    "id_documento": "IdDocumento",
}
rename_map = {}
for alias, std in _ALIAS.items():
    for col in list(_df.columns):
        if col.lower().replace(" ", "") == alias:
            rename_map[col] = std

_df.rename(columns=rename_map, inplace=True)

# Asegurar columnas mínimas
REQUIRED = [
    "NUC", "IdDocumento", "textoPDF",
    "Materia", "Asunto", "TipoFallo"
]
for col in REQUIRED:
    if col not in _df.columns:
        _df[col] = ""

# ── 2. Cliente LLM ─────────────────────────────────────────────────
_client = Together(api_key=TOGETHER_API_KEY)

# ── 3. Funciones internas ──────────────────────────────────────────
_PAT_NUC    = re.compile(r"\b\d{3}-\d{4}-[A-Z]{4}-\d{5}\b", re.I)
_PAT_DOCID  = re.compile(r"\b\d{6,}\b")

def _match_row(msg: str) -> pd.Series | None:
    text = msg.strip()
    # Intentar NUC
    m = _PAT_NUC.search(text)
    if m:
        df2 = _df[_df["NUC"].str.contains(m.group(), na=False)]
        if not df2.empty:
            return df2.iloc[0]
    # Intentar IdDocumento
    m2 = _PAT_DOCID.search(text)
    if m2:
        df2 = _df[_df["IdDocumento"].astype(str).str.contains(m2.group(), na=False)]
        if not df2.empty:
            return df2.iloc[0]
    return None

def run(msg: str) -> str:
    """
    1. Encuentra la fila según NUC o IdDocumento.
    2. Recupera los chunks de FAISS mediante vector similarity.
    3. Le pide a Llama-4 un JSON {resumen, considerandos, fallo_literal}.
    4. Formatea la respuesta Markdown.
    """
    row = _match_row(msg)
    from tools.consulta_doc import _set_active
    _set_active({"metadata": {"NUC": row["NUC"].lower()}})
    print("El numero de caso extraido: ",row["NUC"].lower() )
    if row is None:
        return "⚠️ No encontré ese expediente (NUC) ni IdDocumento en la Juriteca."

    # Recuperar top-1 chunk para contexto
    q_vec = emb.embed_query(row["textoPDF"])
    docs = search_by_vector(q_vec, k=1)
    if not docs:
        return "⚠️ No hay texto disponible para ese documento."

    contenido = docs[0].page_content
    prompt = (
        "Arma un expediente en formato ejecutivo extrayendo cada uno de estos puntos del texto que será proporcionado: "
        f"""
- Numero de tramite : Numero de tramite mencionado en el texto si existe
- Numero de Caso (NUC): {row["NUC"].lower()}
- Parte demandada: Extrae los involucrados demandados en el caso con un formato de listado con Nombre completo y documento de identificación validos (Pasaporte/Cedula de identidad)
- Parte demandante:  Extrae los involucrados demandantes en el caso con un formato de listado con Nombre completo y documento de identificación validos (Pasaporte/Cedula de identidad)
- Representantes de la parte demandada: Extrae los representantes demandados en el caso con un formato de listado con Nombre completo,  documento de identificación validos (Pasaporte/Cedula de identidad) y matricula de abogado (si existe, si no, no mencionar.) 
- Representantes de la parte demandante: Extrae los representantes de los demandantes en el caso con un formato de listado con Nombre completo,  documento de identificación validos (Pasaporte/Cedula de identidad) y matricula de abogado (si existe, si no, no mencionar.) 
- Juez: Nombre completo del juez encargado de la sentencia. 
- Secretario: Secretario encargado de la sentencia.
- Articulos considerados: En los considerandos, extrae el texto donde se mencionan los articulos y desarrolla como fueron ejecutados.
- Fallo: {row["TipoFallo"]} 
Aquí tienes el texto.
        """
        f"{contenido}"
    )

    resp = _client.chat.completions.create(
        model=LLM_MODEL_ID,
        messages=[{"role": "user", "content": prompt}],
        temperature=0.0,
        max_tokens=2200,
    )

    # Formateo Markdown
    return resp.choices[0].message.content