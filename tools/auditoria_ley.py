"""
Audita la sentencia ACTIVA (o la indicada por NUC) y detecta artículos
constitucionales y criterios jurisprudenciales que **podrían** haber sido
citados pero no aparecen. Devuelve JSON + Markdown.

• Citas detectadas con regex + LexNLP-ES.
• Artículos "hermanos" obtenidos de un grafo in-memory.
• Candidatos extra recuperados con law_search() sobre index_laws.
"""

from __future__ import annotations
import json, re
from typing import Dict, List, Tuple

from together import Together
from config import TOGETHER_API_KEY, LLM_MODEL_ID
from vectorstore import law_search
import tools.consulta_doc as cd
from tools.grafo_const import GRAPH_CONST

# LexNLP español (opcional pero recomendado)
try:
    import lexnlp.extract.es.citations as lxc
except ImportError:
    lxc = None  # degrada a regex

# ─────────── parámetros ───────────
client            = Together(api_key=TOGETHER_API_KEY)
SIM_THRESHOLD_LEY = 0.66
K_LAW_CANDIDATES  = 4
MAX_ART_OUT       = 3
MAX_CRIT_OUT      = 3
MAX_TOKENS_LLM    = 1200

# ─────────── extracción de citas ───────────
ART_RX = re.compile(r"\bart[íi]culo[\s·.:]*\s*(\d{1,3})", re.I)

def _extract_cited_articles(text: str) -> set[int]:
    nums = {int(m.group(1)) for m in ART_RX.finditer(text)}
    if lxc:
        try:
            for cit in lxc.get_citations(text):
                if getattr(cit, "page", None):
                    nums.add(int(cit.page))
        except Exception:
            pass
    return nums

# ─────────── helpers ───────────
def _context_sentence(doc_id: str) -> str:
    return "\n\n".join(cd.docs_map.get(doc_id, []))

def _graph_neighbors(cited: set[int]) -> set[int]:
    neigh = set()
    for art in cited:
        if art in GRAPH_CONST:
            neigh.update(GRAPH_CONST.neighbors(art))
    return neigh - cited

# … dentro de tools/auditoria_ley.py …

def _find_omitted_norms(q: str, cited: set[int]) -> Tuple[List[Dict], List[Dict]]:
    pool_art = _graph_neighbors(cited)
    hits = law_search(q, k=K_LAW_CANDIDATES) or []

    art_hits, crit_hits = [], []
    for d in hits:
        src = d.metadata.get("fuente")
        if src == "constitucion":
            num = d.metadata.get("articulo")
            if num and int(num) not in cited:
                pool_art.add(int(num))
                art_hits.append((int(num), d.page_content))   # ← ¡sin [:180]!
        else:
            crit_hits.append({
                "id": str(d.metadata.get("ID") or d.metadata.get("NumDesicion") or "s/d"),
                "resumen": d.page_content                    # ← texto íntegro
            })

    art_omit = []
    for num in sorted(pool_art)[:MAX_ART_OUT]:
        extracto = next((t for n, t in art_hits if n == num), "")
        art_omit.append({
            "articulo": str(num),
            "extracto": extracto                            # ← texto completo
        })

    return art_omit, crit_hits[:MAX_CRIT_OUT]

def _md(rep: Dict) -> str:
    sec = [
        "## Artículos constitucionales posiblemente omitidos",
        *([f"- **Art. {a['articulo']}**: {a['extracto']}" for a in rep["articulos_omitidos"]] or ["—"]),
        "\n## Criterios jurisprudenciales posiblemente omitidos",
        *([f"- **Criterio {c['id']}**: {c['resumen']}" for c in rep["criterios_omitidos"]] or ["—"]),
        "\n## Comentario global",
        rep["comentario_global"],
        ("\n> **Disclaimer:** Los elementos listados _podrían_ ser relevantes. "
         "Su omisión no invalida automáticamente el fallo; debe evaluarse "
         "su pertinencia según el caso concreto.")
    ]
    return "\n".join(sec)

# ─────────── función principal ───────────
def run(user_msg: str) -> str:
    ident = cd.extract_identifier(user_msg)
    if ident:
        cd._set_active({"NUC": ident, "NumeroTramite": ident, "DocumentID": ident})

    if not cd.active_doc:
        return "⚠️ No hay ninguna sentencia activa. Indica el número de caso."

    did = cd._doc_id(cd.active_doc)
    if not did:
        return "⚠️ La sentencia activa no tiene identificador reconocible."

    sent_txt = _context_sentence(did)
    if not sent_txt:
        return "⚠️ Texto de la sentencia no encontrado en memoria."

    cited = _extract_cited_articles(sent_txt)
    art_omit, crit_omit = _find_omitted_norms(user_msg, cited)

    if not art_omit and not crit_omit:
        return "✔️ No se detectaron omisiones normativas relevantes."

    skeleton = {
        "articulos_omitidos": art_omit,
        "criterios_omitidos": crit_omit,
        "comentario_global": (
            "Describe en 3-4 líneas la relevancia de estos artículos/criterios "
            "y cómo su omisión podría afectar la motivación del fallo."
        )
    }

    prompt = (
        "Eres un auditor jurídico experto.\n"
        "Rellena el campo 'comentario_global' y devuelve el JSON en UNA línea:\n"
        f"{json.dumps(skeleton, ensure_ascii=False)}\n\n"
        "===== SENTENCIA =====\n" + sent_txt
    )

    try:
        rsp = client.chat.completions.create(
            model=LLM_MODEL_ID,
            messages=[{"role": "user", "content": prompt}],
            response_format={"type": "json_object"},
            temperature=0.0,
            max_tokens=MAX_TOKENS_LLM,
        )
        data = json.loads(rsp.choices[0].message.content)
        return _md(data)

    except Exception as e:
        return f"⚠️ Error al auditar la sentencia: {str(e)}"
