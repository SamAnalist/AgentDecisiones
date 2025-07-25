# tools/cronologia.py
# =========================================================================
# Query “cronología”: genera la línea de tiempo completa de un caso (NUC)
# =========================================================================
"""
• Detecta el NUC en la pregunta del usuario.
• Recupera **todos** los documentos asociados con `colectar_texto(nuc)`
  (columna `texto_pdf` sin trocear).
• Envía el conjunto al LLM para que produzca:
    1) Una lista JSON ordenada cronológicamente con:
       - fecha         (AAAA-MM-DD o "s/d")
       - tramite       (número o identificador del documento)
       - evento        (resumen de la actuación/decisión)
       - firmantes     (juez, secretario, abogados si aparecen)
    2) Un resumen narrativo de 200-300 palabras.
• Devuelve la respuesta en Markdown con:
    ## Cronología del caso {NUC}
      (tabla)
    ### Resumen narrativo
      (párrafo)
"""
from __future__ import annotations
import re, json, textwrap, pandas as pd
from typing import Dict, List
from together import Together

from config import TOGETHER_API_KEY, LLM_MODEL_ID
from trigger_search_documents import colectar_texto

# ────────────────────────────── LLM ─────────────────────────────────────
_client = Together(api_key=TOGETHER_API_KEY)

# ───────────────────── RegEx para NUC ───────────────────────────────────
_PAT_NUC = re.compile(r"\b\d{3}-\d{4}-[A-Z]{4}-\d{5}\b", re.I)

# ───────────────────── JSON-SCHEMA que el LLM debe respetar ─────────────
_JSON_SCHEMA = """
/*
Estructura requerida:
{
  "timeline":[
     {"fecha":"", "tramite":"", "evento":"", "firmantes":""},
     …
  ],
  "resumen":""
}
- "fecha": AAAA-MM-DD; usar "s/d" si no consta.
- "tramite": número de trámite, auto o sentencia tal como aparece.
- "evento": 1-2 frases que expliquen la actuación o decisión.
- "firmantes": juez, secretario y/o abogados mencionados en el documento.
- "timeline" debe venir ordenado cronológicamente (antiguo→reciente).
- "resumen": 200-300 palabras que integren los hitos principales y
  expliquen el estado actual del expediente.
Devuelve **un único objeto JSON en una sola línea**, sin comentarios extra.
*/
""".strip()

_TEMPLATE = textwrap.dedent("""
Eres un analista jurídico dominicano.
Debes producir la cronología completa del caso usando TODOS los textos
que te daré (uno por documento). Sigue estrictamente el JSON_SCHEMA.

{json_schema}

TEXTOS DE LOS DOCUMENTOS ↓↓↓
""").strip()

# ───────────────────── Helpers de formato Markdown ──────────────────────
def _markdown_table(timeline: List[Dict]) -> str:
    hdr = "| Nº | Fecha | Nº Trámite / Acto | Actuación / decisión | Firmantes |\n" \
          "|----|-------|-------------------|----------------------|-----------|"
    rows = []
    for i, ev in enumerate(timeline, 1):
        rows.append(
            f"| {i} | {ev['fecha'] or 's/d'} | {ev['tramite'] or '—'} | "
            f"{ev['evento'].replace('|','/')} | {ev['firmantes'] or '—'} |"
        )
    return hdr + "\n" + "\n".join(rows)

# ──────────────────────── Función pública ───────────────────────────────
def run(user_msg: str) -> str:
    # 1) Extraer NUC
    m = _PAT_NUC.search(user_msg)
    if not m:
        return ("⚠️ Debes indicar el número de caso (NUC) en el formato "
                "000-0000-AAAA-00000.")
    nuc = m.group(0).lower()

    # 2) Traer documentos completos (dataframe)
    try:
        df_docs: pd.DataFrame = colectar_texto(nuc)
    except Exception as e:
        return f"⚠️ Error al recuperar documentos: {str(e)}"

    if df_docs is None or df_docs.empty:
        return f"⚠️ No se encontraron documentos para el NUC {nuc.upper()}."

    # 2-bis) ACTIVAR sentencia para consulta_doc  ────────────────────────
    #       • Intentamos localizar una fila “principal” (la que venga en el
    #         DataFrame con tipo_fallo o, si no, simplemente la primera).
    try:
        from tools.expediente import _match_row
        row_main = _match_row(user_msg)              # fila exacta (si existe)
    except Exception:
        row_main = None

    if row_main is None or row_main.empty:
        # - fallback al primer documento del DataFrame
        row_main = df_docs.iloc[0]

    from tools.consulta_doc import _set_active
    _set_active({
        "NUC":          str(row_main.get("NUC", nuc)).lower(),
        "NumeroTramite":str(row_main.get("NumeroTramite","")).lower(),
        "DocumentID":   str(row_main.get("IdDocumento","")).lower()
    })

    # 3) Armar corpus completo (un texto por documento)
    textos = []
    for _, r in df_docs.sort_values("FechaCreacion").iterrows():
        tag = f"[{r.get('FechaCreacion','s/d')} | {r.get('NumeroTramite','—')}]"
        textos.append(f"{tag}\n{r['texto_pdf']}")
    corpus = "\n\n".join(textos)

    # 4) Llamar al LLM
    prompt = _TEMPLATE.format(json_schema=_JSON_SCHEMA) + "\n" + corpus
    rsp = _client.chat.completions.create(
        model            = LLM_MODEL_ID,
        messages         = [{"role": "user", "content": prompt}],
        temperature      = 0.0,
        max_tokens       = 4096,
        response_format  = {"type": "json_object"}
    )

    # 5) Parseo y Markdown
    try:
        data     = json.loads(rsp.choices[0].message.content)
        timeline = data["timeline"]
        resumen  = data["resumen"]
    except Exception:
        return "⚠️ El modelo respondió en un formato inesperado."

    md  = f"## Cronología del caso {nuc.upper()}\n\n"
    md += _markdown_table(timeline)
    md += "\n\n---\n\n### Resumen narrativo\n\n" + resumen.strip()
    return md
