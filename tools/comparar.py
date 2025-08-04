# tools/comparar.py  ✨ versión enriquecida con leyes y criterios
import json
from typing import List, Dict
from together import Together
from config import TOGETHER_API_KEY, LLM_MODEL_ID, K_RETRIEVE, SIM_THRESHOLD
from vectorstore import search_by_vector, law_search         # ← law_search añadido
from embed import BNEEmbeddings

# ──────────────────────────────────────────────────────────
client = Together(api_key=TOGETHER_API_KEY)
emb    = BNEEmbeddings()

# ─────────────── ESQUEMA JSON ─────────────────────────────
_JSON_SCHEMA = """
/*
Devuelve UN objeto JSON EN UNA SOLA LÍNEA con las claves:

{
  "casos": [
    {
      "id": "",            // NUC o IdDocumento
      "resumen": "",       // 3-4 líneas
      "considerandos": ""  // 2-3 frases aplicables
    },
    { … segundo caso … }
  ],
  "leyes": [               // máx. 3 artículos constitucionales
    { "articulo": "", "extracto": "" }
  ],
  "criterios": [           // máx. 3 criterios jurisprudenciales
    { "id": "", "resumen": "" }
  ],
  "comparacion": "",       // similitudes / diferencias
  "fallo_sugerido": ""     // motivación final
}
*/
""".strip()

# ─────────────── TEMPLATE DE PROMPT ──────────────────────
_TEMPLATE = """
Eres juez asistente en República Dominicana.
Tarea:
1. Lee la CONSULTA y los CASOS_PARECIDOS.
2. Elige los DOS casos más relevantes.
3. Identifica hasta 3 artículos constitucionales y hasta 3 criterios
   jurisprudenciales que apoyen el fallo.
4. Devuelve el JSON EXACTAMENTE con la estructura indicada.

{json_schema}

========== CONSULTA USUARIO ==========
{pregunta}

========== CASOS PARECIDOS ===========
{contexto}

========== ARTÍCULOS POSIBLES =========
{ctx_leyes}

========== CRITERIOS POSIBLES =========
{ctx_criterios}
""".strip()

# ────────────── Helper Markdown ──────────────────────────
def _md(data: Dict) -> str:
    partes: List[str] = []

    # Casos
    for i, caso in enumerate(data["casos"], 1):
        partes.append(
            f"## Caso {i} ({caso['id']})\n"
            f"- **Resumen:** {caso['resumen']}\n"
            f"- **Considerandos clave:** {caso['considerandos']}"
        )

    # Leyes aplicables
    if data.get("leyes"):
        leyes_md = "\n".join(
            f"- **Art. {l['articulo']}**: {l['extracto']}"
            for l in data["leyes"]
        )
        partes.append(f"## Leyes aplicables\n{leyes_md}")

    # Criterios relevantes
    if data.get("criterios"):
        crit_md = "\n".join(
            f"- **Criterio {c['id']}**: {c['resumen']}"
            for c in data["criterios"]
        )
        partes.append(f"## Criterios relevantes\n{crit_md}")

    # Comparación y fallo
    partes.append(
        "## Comparación y veredicto recomendado\n"
        f"{data['comparacion']}\n\n"
        f"**Fallo sugerido:** {data['fallo_sugerido']}"
    )
    return "\n\n".join(partes)

# ─────────────── FUNCIÓN PRINCIPAL ───────────────────────
def run(msg: str) -> str:
    # 1. Recuperar precedentes
    q_vec = emb.embed_query(msg)
    docs  = search_by_vector(q_vec, k=K_RETRIEVE * 4)
    if not docs:
        return "⚠️ No hallé precedentes relevantes."

    buenos = [d for d in docs if d.metadata.get("score", 1) >= SIM_THRESHOLD]
    if len(buenos) < K_RETRIEVE:
        buenos += [d for d in docs if d not in buenos][:K_RETRIEVE - len(buenos)]

    contexto = "\n\n".join(
        f"[{d.metadata.get('NUC') or d.metadata.get('IdDocumento','s/d')}] "
        + d.page_content[:600]  # recortamos para no exceder tokens
        for d in buenos[:K_RETRIEVE]
    )

    # 2. Recuperar artículos constitucionales y criterios
    legal_hits = law_search(msg, k=8)
    leyes      = [d for d in legal_hits if d.metadata.get("fuente") == "constitucion"][:3]
    criterios  = [d for d in legal_hits if d.metadata.get("fuente") == "criterio"][:3]

    ctx_leyes = "\n".join(
        f"[Art. {l.metadata.get('articulo')}] {l.page_content[:180]}…" for l in leyes
    ) or "—"

    ctx_criterios = "\n".join(
        f"[ID {c.metadata.get('ID')}] {c.page_content[:180]}…" for c in criterios
    ) or "—"

    # 3. Construir prompt
    prompt = _TEMPLATE.format(
        json_schema=_JSON_SCHEMA,
        pregunta=msg.strip(),
        contexto=contexto,
        ctx_leyes=ctx_leyes,
        ctx_criterios=ctx_criterios,
    )

    # 4. Llamar al modelo
    try:
        rsp = client.chat.completions.create(
            model=LLM_MODEL_ID,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.0,
            max_tokens=1500,
            response_format={"type": "json_object"},
        )
        data = json.loads(rsp.choices[0].message.content)
        return _md(data)

    except Exception as e:
        return f"⚠️ Error al generar comparación: {str(e)}"
