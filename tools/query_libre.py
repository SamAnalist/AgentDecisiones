# tools/consulta_concepto.py
"""
Módulo: consulta_concepto
------------------------
Query libre para tu agente RAG legal.  Responde definiciones, conceptos,
procedimientos y aclaraciones jurídicas; mantiene memoria para no
repetir explicaciones y fundamenta sus respuestas en los pasajes
recuperados de tu índice legal.

Requisitos previos en tu stack:
    • law_search.search_with_scores(text, k)  ->  [(Document, score), ...]
    • embed.get_embeddings(name)              ->  Embeddings
    • memory      :  objeto con .get(key), .set(key, value),
                     .last_n_turns(n) ó .summary()
    • variables en config.py:
          SIM_THRESHOLD, LLM_MODEL_ID, TOGETHER_API_KEY
    • pip install together           (o tu proveedor LLM)
"""

from __future__ import annotations
import re
from typing import List

from together import Together               # ↳  sustituye si usas otro LLM
from langchain.text_splitter import RecursiveCharacterTextSplitter

from vectorstore import law_search          # tu wrapper FAISS / Chroma
from embed import get_embeddings
from memory import memory
from config import SIM_THRESHOLD, LLM_MODEL_ID, TOGETHER_API_KEY


# ──────── Config local ────────────────────────────────────────────────────
_CLIENT     = Together(api_key=TOGETHER_API_KEY)
_EMB_LAW    = get_embeddings("laws")
_SPLIT      = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
_MAX_TOKENS = 512
_HISTORY_TURNS = 4            # nº de turnos previos a inyectar como contexto


# ──────── Helper ─────────────────────────────────────────────────────────
def _normalize(term: str) -> str:
    """Quita caracteres no alfabéticos y normaliza a minúsculas."""
    return re.sub(r"[^A-Za-zÁ-Úá-úÜüÑñ ]+", "", term).strip().lower()


# ──────── API pública ────────────────────────────────────────────────────
def query_libre_run(question: str,
                      k: int = 6,
                      temperature: float = 0.2) -> str:
    """
    Responde preguntas del tipo «¿Qué es la litispendencia?».

    ▸ Si la definición ya se dio antes en esta conversación, se devuelve tal
      cual (hace de mini-glosario en memoria).
    ▸ Si no, recupera pasajes legales relevantes y consulta al LLM.
    ▸ Retorna un texto fundamentado y cita artículos si están en el contexto.
    """
    # 1) buscar en memoria cacheada
    term_key = _normalize(question)
    prev = memory.get(f"def:{term_key}")
    if prev:
        return prev

    # 2) recuperación de contexto legal
    docs_scores = law_search.search_with_scores(question, k=k)
    ctx_passages: List[str] = [
        d.page_content for d, s in docs_scores if s < SIM_THRESHOLD
    ]
    context = "\n\n".join(ctx_passages) or "NO_MATCH"

    # 3) prompt dinámico para el LLM
    history = memory.last_n_turns(_HISTORY_TURNS)

    system_prompt = (
        "Eres un abogado experto en derecho dominicano. "
        "Responde preguntas jurídicas en lenguaje claro, cita normas "
        "y evita repetir definiciones ya dadas previamente. "
        "Si no existe respaldo legal sólido, reconoce esa limitación."
    )

    user_prompt = (
        f"Historial reciente:\n{history}\n\n"
        f"Pregunta actual: {question}\n\n"
        f"Contexto legal recuperado:\n{context}\n\n"
        "Instrucciones de formato:\n"
        "• Comienza con una definición directa.\n"
        "• Incluye al menos una cita normativa o jurisprudencial si procede.\n"
        "• Sé conciso (máx. 250 palabras).\n"
        "• Si el contexto = 'NO_MATCH', di explícitamente que no se encontró "
        "soporte normativo suficiente.\n"
    )

    resp = _CLIENT.chat.completions.create(
        model       = LLM_MODEL_ID,
        messages    = [
            {"role": "system", "content": system_prompt},
            {"role": "user",   "content": user_prompt}
        ],
        max_tokens  = _MAX_TOKENS,
        temperature = temperature
    ).choices[0].message.content.strip()

    # 4) guardar en memoria para futuras consultas
    memory.set(f"def:{term_key}", resp)
    return resp


