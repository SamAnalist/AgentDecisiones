# tools/comparar_ids.py
# -----------------------------------------------------------
import re, json, textwrap
from typing import Tuple
from together import Together
import pandas as pd
from config import DATA_DIR, TOGETHER_API_KEY, LLM_MODEL_ID

# ─── Cargar DataFrame con tus sentencias (mismo XLSX) ───────
_df = pd.read_excel(DATA_DIR / "output (1).xlsx")
_df["NUC"] = _df["NUC"].astype(str).str.lower()
_df["IdDocumento"] = _df["IdDocumento"].astype(str).str.lower()

# ─── Cliente LLM ────────────────────────────────────────────
client = Together(api_key=TOGETHER_API_KEY)

# ─── Prompt plantilla ──────────────────────────────────────
_TEMPLATE = textwrap.dedent("""
Compara las dos sentencias y devuelve un JSON EN UNA SOLA LÍNEA con
la siguiente estructura:

{{
  "comparacion": "",   // similitudes y diferencias clave
  "conclusion": "" // Escribe una conclusión de los fallos y considerandos de los documentos.
}}

========== DOCUMENTO A ({id_a}) ==========
{text_a}

========== DOCUMENTO B ({id_b}) ==========
{text_b}
""").strip()

MAX_CHARS = 40_000     # límite para cada texto en el prompt

# ─── Helpers ───────────────────────────────────────────────
_PAT_NUC    = re.compile(r"\b\d{3}-\d{4}-[A-Z]{4}-\d{5}\b", re.I)
_PAT_DOCID  = re.compile(r"\b\d{6,}\b")

def _extract_two_ids(msg: str) -> Tuple[str, str] | None:
    """Devuelve los dos primeros ids encontrados en minúsculas."""
    ids = []

    # 1) NUCs
    ids += _PAT_NUC.findall(msg)
    # 2) IdDocumento numérico largo
    ids += _PAT_DOCID.findall(msg)

    ids = [i.lower() for i in ids]
    uniq = []
    for i in ids:
        if i not in uniq:
            uniq.append(i)
    return (uniq[0], uniq[1]) if len(uniq) >= 2 else None

def _get_text(doc_id: str) -> str | None:
    """Busca el texto correspondiente a un NUC o IdDocumento."""
    row = _df[_df["NUC"].str.contains(doc_id, na=False)]
    if row.empty:
        row = _df[_df["IdDocumento"].str.contains(doc_id, na=False)]
    if row.empty:
        return None
    return row.iloc[0]["textoPDF"]

# ─── Función pública ───────────────────────────────────────
def run(msg: str) -> str:
    """
    Compara dos sentencias de la base de datos dadas por sus
    identificadores (NUC o IdDocumento).
    """
    ids = _extract_two_ids(msg)
    if not ids:
        return "⚠️ No hallé precedentes relevantes en la base para esa consulta."

    id_a, id_b = ids
    text_a = _get_text(id_a)
    text_b = _get_text(id_b)

    if not text_a or not text_b:
        return "⚠️ No encontré uno o ambos identificadores en la base de datos."

    prompt = _TEMPLATE.format(
        id_a=id_a, id_b=id_b,
        text_a=text_a[:MAX_CHARS],
        text_b=text_b[:MAX_CHARS],
    )

    try:
        rsp = client.chat.completions.create(
            model=LLM_MODEL_ID,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.0,
            max_tokens=800,
            response_format={"type": "json_object"},
        )
        data = json.loads(rsp.choices[0].message.content)
        return (
            "## Comparación de sentencias\n\n"
            f"{data['comparacion']}\n\n"
            f"**Fallo sugerido:** {data['conclusion']}"
        )

    except Exception as e:
        return f"⚠️ Error al comparar sentencias: {str(e)}"
