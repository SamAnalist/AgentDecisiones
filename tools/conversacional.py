"""
tools/conversacional.py
-----------------------
Fallback genérico: responde preguntas libres consultando simultáneamente

  • Base de leyes      → law_search()
  • Base de sentencias → search_by_vector()
  • Web                → DuckDuckGoSearchAPIWrapper.results()

Devuelve Markdown con citas numeradas [n] que el LLM genera
a partir de los fragmentos suministrados en el prompt.
"""
from __future__ import annotations
import logging, json, textwrap
from typing import List, Dict, Any

from together import Together
from langchain_community.utilities import DuckDuckGoSearchAPIWrapper

from config       import TOGETHER_API_KEY, LLM_MODEL_ID, K_RETRIEVE
from vectorstore  import law_search, search_by_vector
from embed        import BNEEmbeddings
from memory import memory
history = memory.load_memory_variables({})
# ────────────────────── inicialización ──────────────────────
log     = logging.getLogger(__name__)
client  = Together(api_key=TOGETHER_API_KEY)
emb     = BNEEmbeddings()
_DUCK   = DuckDuckGoSearchAPIWrapper()   # motor web

# ───────────────────── helpers ──────────────────────
def _collect_context(
    question: str,
    k_cases: int = 4,
    k_laws:  int = 4,
    k_web:   int = 3
) -> List[Dict[str, Any]]:
    """Busca contexto híbrido (sentencias, leyes, web) y lo devuelve normalizado."""
    ctx: List[Dict[str, Any]] = []

    # 1) Sentencias (vector similarity)
    q_vec = emb.embed_query(question)
    hits  = search_by_vector(q_vec, k=k_cases)
    for d in hits:
        ctx.append({
            "type": "case",
            "id":   d.metadata.get("NUC") or d.metadata.get("IdDocumento") or "s/d",
            "text": d.page_content[:2000]          # recorte para tokens
        })

    # 2) Leyes / criterios
    for d in law_search(question, k=k_laws):
        src = d.metadata.get("fuente")
        if src == "constitucion":
            tag = f"Art. {d.metadata.get('articulo')}"
        elif src == "criterio":
            tag = f"Crit. {d.metadata.get('ID') or d.metadata.get('NumDecision')}"
        else:
            tag = src or "ley"
        ctx.append({"type": "law", "id": tag, "text": d.page_content})

    # 3) Web snippets (DuckDuckGo)
    try:
        web_hits = _DUCK.results(f"{question} República Dominicana", k_web)
    except Exception as e:
        log.warning("DuckDuckGoSearch error: %s", e)
        web_hits = []

    for hit in web_hits:
        snippet = (hit.get("body") or hit.get("title") or "").strip()
        if not snippet:
            continue
        ctx.append({
            "type": "web",
            "id":   hit.get("href", "url-desconocida"),
            "text": snippet[:300]               # recorte
        })

    return ctx


def _block_for_prompt(ctx: List[Dict[str, Any]]) -> str:
    """Renderiza cada fragmento como `[n] (<id>) <texto>`"""
    return "\n".join(
        f"[{i}] ({c['id']}) {c['text']}" for i, c in enumerate(ctx, 1)
    )

# ───────────────────── API principal ──────────────────────
def run(
    msg: str,
    *,
    k_cases: int = 4,
    k_laws:  int = 4,
    k_web:   int = 3
) -> str:
    """
    Conversación libre con recuperación híbrida.
    """
    # 1) Buscar contexto
    ctx         = _collect_context(msg, k_cases, k_laws, k_web)
    refs_block  = _block_for_prompt(ctx)

    # 2) Prompt
    system = (
        "Eres un asistente jurídico dominicano. "
        "Si la pregunta no es estrictamente legal, contéstala igualmente con la mejor "
        "respuesta posible, citando siempre las fuentes entre corchetes."
        f"Este es el historial de mensajes hasta ahora {history}"
    )
    user = textwrap.dedent(f"""
        Pregunta del usuario:
        {msg}

        Fuentes disponibles:
        {refs_block}

        Instrucciones:
        • Responde en un solo bloque Markdown.
        • Usa las referencias [n] cuando cites un dato concreto.
        • Si no sabes algo, admítelo.
    """).strip()

    log.debug("Prompt conversacional:\n%s", user[:1000])
    rsp = client.chat.completions.create(
        model=LLM_MODEL_ID,
        messages=[
            {"role": "system", "content": system},
            {"role": "user",   "content": user},
        ],
        temperature=0.2,
        max_tokens=1400,
    )
    return rsp.choices[0].message.content.strip()


__all__ = ["run"]
