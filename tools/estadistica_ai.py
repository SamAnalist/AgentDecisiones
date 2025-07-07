# tools/estadistica_ai.py
"""
Responde preguntas estadísticas abiertas, p. ej.:
  • ¿Cuántos casos de robo hay?
  • Promedio de indemnizaciones por homicidio en 2023
Proceso:
  1. NL → 1 línea de código pandas (Together AI, temperatura 0).
  2. Si el código referencia columnas existentes, se ejecuta en sandbox.
  3. Si devuelve "__fallback__", hace búsqueda semántica en FAISS
     con sinónimos básicos y deduplica por NUC / IdDocumento.
"""
from __future__ import annotations
import re, ast, pandas as pd

import streamlit
from langchain_core.tools import Tool
from langchain_experimental.utilities import PythonREPL
from together import Together
from vectorstore import vectordb              # índice FAISS ya cargado
from tools.consulta_doc import _doc_id        # helper para extraer id del chunk
from config import DATA_DIR, TOGETHER_API_KEY, LLM_MODEL_ID

# ─────────────────── recursos en memoria ──────────────────────────────
df   = pd.read_excel(DATA_DIR / "output (1).xlsx")   # ajusta ruta si cambia
# ← inicializa REPL con el DataFrame:
repl = PythonREPL(locals={"df": df})
t_client = Together(api_key=TOGETHER_API_KEY)

# ─────────────────── NL → código pandas ───────────────────────────────
PROMPT_NL2CODE = (
    "Eres experto en pandas. Devuelve UNA sola línea de código Python que se "
    "ejecute sobre un DataFrame llamado df y responda la PREGUNTA. "
    "No añadas comentarios ni explicación. "
    "Si df no tiene las columnas necesarias, responde únicamente '__fallback__'."
)

def nl_to_code(question: str) -> str:
    resp = t_client.chat.completions.create(
        model       = LLM_MODEL_ID,
        messages    = [
            {"role": "system", "content": PROMPT_NL2CODE},
            {"role": "user",   "content": question},
        ],
        temperature = 0.0,
        max_tokens  = 180,
    )
    return resp.choices[0].message.content.strip()

# ─────────────────── fallback semántico ───────────────────────────────
def _semantic_count(term: str, k: int = 1000) -> int:
    """Cuenta expedientes cuyo embedding se parece al término."""
    # Sinónimos rápidos; amplía según tu dominio
    synonym_map = {
        "robo": ["hurto", "sustracción"],
        "homicidio": ["asesinato"]
    }
    terms = [term] + synonym_map.get(term, [])
    doc_ids = set()
    for t in terms:
        hits = vectordb.similarity_search(t, k=k)
        doc_ids |= {_doc_id(h) for h in hits if _doc_id(h)}
    return len(doc_ids)

# ─────────────────── parsing helpers ──────────────────────────────────
def _extract_code(raw: str) -> str:
    """Si el modelo envolvió en ```python ...```, extrae el bloque."""
    m = re.search(r"```python(.*?)```", raw, re.S)
    return m.group(1).strip() if m else raw.strip()

def _extract_term(msg: str) -> str:
    m = re.search(r"casos?\s+de\s+([\w\s]+)", msg, re.I)
    return m.group(1).lower().strip() if m else "caso"

# ─────────────────── API principal ────────────────────────────────────
def run(msg: str) -> str:
    code = _extract_code(nl_to_code(msg))
    import streamlit
    streamlit.write(code)
    # validar sintaxis
    try:
        ast.parse(code, mode="exec")
    except SyntaxError:
        code = "__fallback__"

    # Camino A: ejecutar código pandas
    if code != "__fallback__":
        try:
            result = repl.run(code)          # sandbox de langchain
            return f"Resultado: **{result}**"
        except Exception:
            code = "__fallback__"            # cae al semántico

    # Camino B: búsqueda semántica
    term   = _extract_term(msg)
    total  = _semantic_count(term)
    return f"En la base hay **{total}** sentencias que mencionan {term}."
