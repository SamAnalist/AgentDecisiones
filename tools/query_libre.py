# tools/query_libre.py  — v2025-07-30
from __future__ import annotations
import re, json
from pathlib import Path
from typing import List, Optional

from together import Together
from langchain_community.utilities import DuckDuckGoSearchAPIWrapper
from langchain.memory import ConversationBufferMemory

from config import DATA_DIR, LLM_MODEL_ID, TOGETHER_API_KEY
from vectorstore import law_search                           # ← ya carga index_laws

# ──────────────── LLM & search wrappers ───────────────────
_CLIENT          = Together(api_key=TOGETHER_API_KEY)
_DUCK            = DuckDuckGoSearchAPIWrapper()

_MAX_TOKENS      = 512
_HISTORY_TURNS   = 4
_SEARCH_TOP_K    = 20
_CRIT_TOP_K      = 5

memory: ConversationBufferMemory = ConversationBufferMemory(return_messages=False)

# ──────────────── Carga Constitución en dict ──────────────
CONST_PATH = Path(DATA_DIR) / "constitucion.csv"
_const_map: dict[int, str] = {}
if CONST_PATH.exists():
    import pandas as pd
    df_const = pd.read_csv(CONST_PATH)
    for _, row in df_const.iterrows():
        try:
            _const_map[int(row["ArticuloNo"])] = str(row["ArticuloContenido"]).strip()
        except (ValueError, KeyError):
            continue  # skip malformed rows


# ──────────────── Helpers generales ──────────────────────
_ART_RX = re.compile(
    r"""art[ií]culo\s+        # palabra artículo
        (?P<num>\d{1,3})      # número (1-3 dígitos)
        (?:\s*(?:de|del)?\s*
        (?:la\s+)?constituci[oó]n)   # opcional "de la constitución"
    """, re.I | re.X,
)

def _extract_articulo_num(q: str) -> Optional[int]:
    m = _ART_RX.search(q.lower())
    if m:
        try:
            return int(m.group("num"))
        except ValueError:
            return None
    return None

def _duck_snippets(q: str) -> str:
    raw   = _DUCK.run(f"{q} República Dominicana derecho")
    return "\n".join(raw.split("\n")[:_SEARCH_TOP_K])

def _legal_snippets(q: str, k: int = _CRIT_TOP_K) -> str:
    hits = law_search(q, k=k) or []
    partes: List[str] = []
    for d in hits:
        src = d.metadata.get("fuente")
        tag = f"Art. {d.metadata.get('articulo')}" if src == "constitucion" \
              else f"Criterio {d.metadata.get('ID')}"
        partes.append(f"[{tag}] {d.page_content[:180]}…")
    return "\n".join(partes) if partes else "—"

def _last_turns(hist: str, n: int) -> str:
    lines = [l for l in hist.strip().split("\n") if l]
    return "\n".join(lines[-n * 2:])           # user+assistant ≈ 2 líneas/t

# ──────────────── Respuesta directa de la Constitución ───
def _answer_constitution(article_num: int) -> str:
    texto = _const_map.get(article_num)
    if not texto:
        return ""
    return (
        f"**Artículo {article_num} — Constitución de la República Dominicana**\n\n"
        f"{texto}\n\n"
        "_Fuente: Constitución (G.O. 10805-10-06-2015)._"
    )

# ──────────────── Entrada principal ──────────────────────
def query_libre_run(question: str,
                    temperature: float = 0.0) -> str:
    # 0) ¿Pregunta directa a la Constitución?
    art_num = _extract_articulo_num(question)
    if art_num is not None:
        respuesta = _answer_constitution(art_num)
        if respuesta:
            # guarda en memoria y devuelve
            memory.save_context({"input": question}, {"output": respuesta})
            return respuesta

    # 1) Historial reciente
    hist_str = _last_turns(memory.buffer, _HISTORY_TURNS)

    # 2) Snippets externos
    web_snips   = _duck_snippets(question)
    leyes_snips = _legal_snippets(question)

    # 3) Prompt al LLM
    system = (
        "Eres un jurista experto en derecho dominicano. "
        "Responde con rigor y brevedad (≤250 palabras). "
        "Cuando cites normas, indica su número de artículo "
        "o identificador de criterio."
    )
    user = json.dumps({
        "historial": hist_str,
        "pregunta": question,
        "web": web_snips,
        "base_normativa": leyes_snips,
    }, ensure_ascii=False, indent=2)

    respuesta = _CLIENT.chat.completions.create(
        model       = LLM_MODEL_ID,
        messages    = [
            {"role": "system", "content": system},
            {"role": "user",   "content": user},
        ],
        max_tokens  = _MAX_TOKENS,
        temperature = temperature,
    ).choices[0].message.content.strip()

    # 4) Persistir memoria
    memory.save_context({"input": question}, {"output": respuesta})
    return respuesta


__all__ = ["query_libre_run", "memory"]
