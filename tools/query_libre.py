# tools/query_libre.py  (versión 100 % web-augmented)

from __future__ import annotations
import re
from typing import List

from together import Together
from langchain_community.utilities import DuckDuckGoSearchAPIWrapper
from langchain.memory import ConversationBufferMemory

from config import LLM_MODEL_ID, TOGETHER_API_KEY

# ───── Config ─────
_CLIENT          = Together(api_key=TOGETHER_API_KEY)
_MAX_TOKENS      = 512
_HISTORY_TURNS   = 4        # nº de turnos previos
_SEARCH_TOP_K    = 20        # líneas máximas a inyectar
_DUCK            = DuckDuckGoSearchAPIWrapper()

memory: ConversationBufferMemory = ConversationBufferMemory(return_messages=False)

# ───── Helpers ─────
def _normalize(txt: str) -> str:
    return re.sub(r"[^A-Za-zÁ-Úá-úÜüÑñ ]+", "", txt).strip().lower()

def _last_n_turns(buf: str, n: int) -> str:
    lines = [l for l in buf.strip().split("\n") if l]
    return "\n".join(lines[-n*2:])

def _duck_snippets(q: str) -> str:
    raw = _DUCK.run(f"{q} definición jurídica República Dominicana")
    # primer corte por líneas
    cuts = raw.split("\n")[:_SEARCH_TOP_K]
    # segundo corte por longitud total
    joined = "\n".join(cuts)
    print(joined)
    return joined

# ───── Consulta libre ─────
def query_libre_run(question: str,
                    temperature: float = 0.0) -> str:
    """
    Siempre realiza una búsqueda DuckDuckGo y usa el historial de
    conversación para responder preguntas jurídicas breves.
    """
    # 1) glosario
    mem_vars  = memory.load_memory_variables({})
    glossary  = mem_vars.get("definitions", {})
    term_key  = _normalize(question)

    # 2) historial y snippets web (siempre)
    history_str  = _last_n_turns(memory.buffer, _HISTORY_TURNS)
    web_snip     = _duck_snippets(question)

    # 3) prompt
    system_prompt = (
        "Eres un abogado experto en derecho dominicano. "
        "Responde en un español claro y profesional, máximo 250 palabras. "
        "Cita la base normativa si es conocida. "
        "Si la información del fragmento web es insuficiente o dudosa, indícalo."
    )

    user_prompt = (
        f"Historial reciente:\n{history_str}\n\n"
        f"Pregunta del usuario: {question}\n\n"
        f"Fuente web (DuckDuckGo):\n{web_snip}\n\n"
    )

    # 4) LLM
    resp = _CLIENT.chat.completions.create(
        model       = LLM_MODEL_ID,
        messages    = [
            {"role": "system", "content": system_prompt},
            {"role": "user",   "content": user_prompt},
        ],
        max_tokens  = _MAX_TOKENS,
        temperature = temperature,
    ).choices[0].message.content.strip()

    # 5) guardar en memoria
    glossary[term_key] = resp
    memory.save_context(
        {"input": question},
        {"output": resp, "definitions": glossary}
    )
    return resp

__all__ = ["query_libre_run", "memory"]
