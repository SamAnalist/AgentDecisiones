# tools/comparar.py  ✨ output enriquecido
import json
from typing import List, Dict
from together import Together
from config import TOGETHER_API_KEY, LLM_MODEL_ID, K_RETRIEVE, SIM_THRESHOLD
from vectorstore import search_by_vector
from embed import BNEEmbeddings

# ──────────────────────────────────────────────────────────
client = Together(api_key=TOGETHER_API_KEY)
emb    = BNEEmbeddings()

# ────────────── plantilla JSON y prompt ───────────────────
_JSON_SCHEMA = """
/*
Devuelve UN objeto JSON EN UNA SOLA LÍNEA con las claves:

{
  "casos": [
    {
      "id":     "",   // NUC o IdDocumento del precedente elegido
      "resumen": "",  // 3-4 líneas: hechos, pretensión y resultado
      "considerandos": "" // 2-3 frases con los fundamentos que
                          // podrían aplicarse al caso del usuario
    },
    { ... segundo caso ... }
  ],
  "comparacion": "",  // síntesis de similitudes/diferencias
  "fallo_sugerido": ""// Toma los considerandos y fallo de ambos casos y determina cuales articulos podrían aplicar en el caso dado por el usuario. 
}

• Si el texto fuente no muestra NUC toma “s/d”.
*/
""".strip()

_TEMPLATE = """
Eres juez asistente en República Dominicana.
Tarea:
1. Lee la CONSULTA del usuario y los CASOS_PARECIDOS que siguen.
2. Elige los DOS casos más relevantes.
3. Rellena el JSON exactamente como se describe.

{json_schema}

================= CONSULTA USUARIO =================
{pregunta}

================= CASOS_PARECIDOS ==================
{contexto}
""".strip()

# ─────────── helpers markdown ──────────────────────
def _md(data: Dict) -> str:
    secciones = []
    for idx, caso in enumerate(data["casos"], 1):
        secciones.append(
            f"## Caso {idx} ({caso['id']})\n"
            f"- **Resumen:** {caso['resumen']}\n"
            f"- **Considerandos clave:** {caso['considerandos']}"
        )
    secciones.append(
        "## Comparación y veredicto recomendado\n"
        f"{data['comparacion']}\n\n"
        f"**Fallo sugerido:** {data['fallo_sugerido']}"
    )
    return "\n\n".join(secciones)

# ─────────── función principal ─────────────────────
def run(msg: str) -> str:
    q_vec = emb.embed_query(msg)
    docs  = search_by_vector(q_vec, k=K_RETRIEVE * 4)
    if not docs:
        return "⚠️ No hallé precedentes relevantes."

    buenos = [d for d in docs if d.metadata.get("score", 1) >= SIM_THRESHOLD]
    if len(buenos) < K_RETRIEVE:
        buenos += [d for d in docs if d not in buenos][:K_RETRIEVE - len(buenos)]

    contexto = "\n\n".join(
        f"[{d.metadata.get('NUC') or d.metadata.get('IdDocumento','s/d')}] "
        + d.page_content  # recorta para no pasar el límite
        for d in buenos[:K_RETRIEVE]
    )

    prompt = _TEMPLATE.format(
        json_schema=_JSON_SCHEMA,
        pregunta=msg.strip(),
        contexto=contexto
    )

    try:
        rsp = client.chat.completions.create(
            model=LLM_MODEL_ID,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.0,
            max_tokens=1500,
            response_format={"type": "json_object"}
        )
        data = json.loads(rsp.choices[0].message.content)
        return _md(data)

    except Exception as e:
        return f"⚠️ Error al generar comparación: {str(e)}"
