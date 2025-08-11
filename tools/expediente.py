"""
tools/expediente.py
-------------------
Herramienta “expediente”: busca un NUC o IdDocumento,
recupera los chunks relacionados y genera un JSON con:
  - resumen
  - considerandos
  - fallo_literal
"""
import re, textwrap
import json
import pandas as pd
from together import Together
from config import DATA_DIR, TOGETHER_API_KEY, LLM_MODEL_ID
from vectorstore import search_by_vector
# ── 1. Cargar y normalizar DataFrame ───────────────────────────────
_df = pd.read_excel(DATA_DIR / "output (1).xlsx")
from typing import Dict, List

# Normalizar nombres: quitar espacios y mantener mayúsculas/minúsculas exactas
_df.columns = [c.strip() for c in _df.columns]
# ── NUEVO JSON_SCHEMA AUTODOCUMENTADO ──────────────────────────────
def _match_row(msg:str):
    m = _PAT_NUC.search(msg)
    if m is not None:
        d = _df[_df["NUC"].str.contains(m.group(), na=False)]
        if not d.empty: return d.iloc[0]
    m = _PAT_DOCID.search(msg)
    if m is not None:
        d = _df[_df["IdDocumento"].astype(str).str.contains(m.group(), na=False)]
        if not d.empty: return d.iloc[0]
    return None

def _md(data:Dict) -> str:
    db  = data["datos_basicos"]
    prt = data["partes"]
    fun = data["funcionarios"]

    def _tbl(lst:List[Dict], hdr:str)->str:
        if not lst: return "- —"
        rows = [f"- **{p['nombre']}** (ID: {p.get('doc_id','—')}"
                + (f", matr. {p['matricula']})" if p.get("matricula") else ")" )
                for p in lst if p.get("nombre")]
        return "\n".join(rows)

    md  = "### 1. Datos básicos\n"
    md += "\n".join(f"- **{k.replace('_',' ').capitalize()}**: {v}"
                    for k,v in db.items() if v)

    md += "\n\n### 2. Partes\n**Demandantes:**\n"   + _tbl(prt["demandantes"],"") + \
          "\n\n**Demandados:**\n"                  + _tbl(prt["demandados"],"") + \
          "\n\n**Repr. demandantes:**\n"           + _tbl(prt["repr_demandantes"],"") + \
          "\n\n**Repr. demandados:**\n"            + _tbl(prt["repr_demandados"],"")

    md += "\n\n### 3. Funcionarios\n"
    md += f"- **Juez:** {fun.get('juez','—')}\n- **Secretario:** {fun.get('secretario','—')}"

    md += "\n\n### 4. Artículos considerados\n" + (data["articulos_considerados"] or "—")
    md += "\n\n### 5. Resumen ejecutivo\n"      + (data["resumen"] or "—")
    md += "\n\n### 6. Considerandos\n"          + (data["considerandos"] or "—")
    md += "\n\n### 7. Fallo literal\n"          + (data["fallo_literal"] or "—")
    return md

_JSON_SCHEMA = """
/*
Explicándole al modelo:

- "datos_basicos":  identifica el expediente.
    • nuc              → Número Único de Caso (formato 000-0000-AAAA-00000).
    • numero_tramite   → Identificador del trámite dentro del caso, si aparece.
    • materia          → Rama del derecho (civil, penal, laboral…).
    • asunto           → Naturaleza específica (ej.: reparación de daños).
    • tipo_fallo       → Clase de decisión (condenatoria, inadmisoria…).
    • fecha_decision   → Fecha en formato AAAA-MM-DD, si figura en el texto.

- "partes":  actores procesales.
    • demandantes / demandados            → lista de personas o entidades con su documento de identidad (cédula o pasaporte) si se menciona.
    • repr_demandantes / repr_demandados  → apoderados de cada parte; incluye matrícula de abogado si la sentencia la muestra.

- "funcionarios":
    • juez        → nombre completo del juez o presidente del tribunal colegiado.
    • secretario  → secretario que firma el fallo.

- "articulos_considerados":  copia literal (o parafraseada brevemente) de los artículos de ley o de la Constitución citados en los considerandos y cómo se aplican.

- "resumen":  5-8 líneas con hechos relevantes y pretensión.
- "considerandos":  motivación jurídica (valoración de la prueba + norma aplicada).
- "fallo_literal":  transcripción exacta de la parte dispositiva (todo lo que sigue a “FALLA”).

DEVUELVE ÚNICAMENTE el objeto JSON que sigue, sin comentarios ni saltos de línea extra.
*/

{"datos_basicos":{"nuc":"","numero_tramite":"","materia":"","asunto":"",
"tipo_fallo":"","fecha_decision":""},"partes":{"demandantes":[{"nombre":"",
"doc_id":""}],"demandados":[{"nombre":"","doc_id":""}],
"repr_demandantes":[{"nombre":"","doc_id":"","matricula":""}],
"repr_demandados":[{"nombre":"","doc_id":"","matricula":""}]},
"funcionarios":{"juez":"","secretario":""},"articulos_considerados":"",
"resumen":"","considerandos":"","fallo_literal":""}
""".strip()
# ─── Prompt plantilla ───────────────────────────────────────────────
_TEMPLATE = textwrap.dedent("""
Eres un analista jurídico. Devuelve **un solo JSON en UNA línea** con la
estructura indicada (no añadas saltos de línea). Extrae los nombres tal como
aparecen en el texto, completa doc_id/matrícula cuando existan y deja string
vacío si no aparece.

{json_schema}

TEXTO ↓↓↓
""").strip()
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
    if row is None:
        return "⚠️ No encontré ese expediente (NUC) ni IdDocumento en la Juriteca."

    from tools.consulta_doc import _set_active
    # ── activar sentencia para consulta_doc ────────────────────────────
    _set_active({
        "NUC": str(row["NUC"]).lower(),
        "NumeroTramite": str(row.get("NumeroTramite", "")).lower(),
        "DocumentID": str(row.get("IdDocumento", "")).lower()
    })
    print("El numero de caso extraido: ",row["NUC"].lower() )
    if row is None:
        return "⚠️ No encontré ese expediente (NUC) ni IdDocumento en la Juriteca."

    # Recuperar top-1 chunk para contexto
    texto = row["textoPDF"]
    prompt = _TEMPLATE.format(json_schema=_JSON_SCHEMA) + "\n" + texto

    rsp = _client.chat.completions.create(
        model=LLM_MODEL_ID,
        messages=[{"role": "user", "content": prompt}],
        temperature=0, max_tokens=2200,
        response_format={"type": "json_object"}
    )
    try:
        data = json.loads(rsp.choices[0].message.content)
    except json.JSONDecodeError:
        return "⚠️ Error al interpretar la respuesta del modelo."

    return _md(data)