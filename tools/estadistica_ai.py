"""
tools/estadistica_ai.py  —  Conteo semántico sin crear archivos nuevos
 Usa FAISS directo + LLM para sinónimos y verificación zona gris
"""
import sys, os, pathlib, re, math
from concurrent.futures import ThreadPoolExecutor
from functools import lru_cache
from typing import List, Set

ROOT = pathlib.Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from config import (
    TOGETHER_API_KEY, LLM_MODEL_ID,
    EMBED_MODEL_ID, SIM_THRESHOLD_est, GREY_MARGIN,
)
from embed import BNEEmbeddings
from semantic_search import search_with_scores
from together import Together

# ─── LLM & embedder ─────────────────────────────────────────────────
embedder = BNEEmbeddings()
llm      = Together(api_key=TOGETHER_API_KEY)

# ─── Helpers IDs ────────────────────────────────────────────────────
def _unique_id(meta: dict | None) -> str | None:
    if not meta:
        return None
    for k in ("DocumentID", "IdDocumento", "NUC", "NumeroTramite"):
        if meta.get(k):
            return str(meta[k]).lower()
    return None

# ─── Concepto + sinónimos LLM (cache) ───────────────────────────────
def _extract_concept(question: str) -> str:
    prompt = (
        f"Devuelve el concepto jurídico principal de la pregunta en ≤6 palabras, todo en minúsculas, sin saltos de línea ni explicación:\n{question}"
    )
    rsp = llm.chat.completions.create(
        model=LLM_MODEL_ID,
        messages=[{"role": "user", "content": prompt}],
        temperature=0.0,
        max_tokens=20
    )
    print(rsp.choices[0].message.content.lower().strip())
    return rsp.choices[0].message.content.lower().strip()

@lru_cache(maxsize=256)
def _expand_terms(base: str) -> List[str]:
    prompt = f"Da hasta 5 sinónimos jurídicos en minúsculas, separados solo por coma, sin textos extra. Ejemplo: robo, hurto, sustracción\nTérmino: {base}"
    rsp = llm.chat.completions.create(
        model=LLM_MODEL_ID,
        messages=[{"role": "user", "content": prompt}],
        temperature=0.0,
        max_tokens=20,
    )
    syns = [s.strip().lower() for s in rsp.choices[0].message.content.split(",")]
    print(syns)
    return list({base.lower(), *syns})

# ─── FAISS búsqueda + filtrado ──────────────────────────────────────
def _ids_for_term(term: str, top_k: int = 800):
    vec = embedder.embed_query(term)
    hits = search_with_scores(vec, k=top_k)

    ids_ok, grey = set(), []
    lower = SIM_THRESHOLD_est
    upper = lower + GREY_MARGIN

    for doc, dist in hits:
        uid = _unique_id(doc.metadata)
        if not uid:
            continue
        if dist <= lower:
            ids_ok.add(uid)
            if dist > lower - GREY_MARGIN:
                grey.append((doc, dist))
    return ids_ok, grey

# ─── Verificación zona gris LLM ─────────────────────────────────────
def _verify_docs(concept: str, grey_hits: list, batch: int = 20) -> Set[str]:
    if not grey_hits:
        return set()
    accepted = set()
    for i in range(0, len(grey_hits), batch):
        chunk = grey_hits[i : i + batch]
        bullets = [
            f"{idx+1}) {doc.page_content[:300].replace(chr(10),' ')}"
            for idx, (doc, _) in enumerate(chunk)
        ]
        prompt = (
            f"Concepto: «{concept}». Indica S/N si el fragmento se relaciona.\n"
            + "\n".join(bullets)
        )
        rsp = llm.chat.completions.create(
            model=LLM_MODEL_ID,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.0,
            max_tokens=len(chunk) * 2,
        )
        answers = re.findall(r"[SN]", rsp.choices[0].message.content.upper())
        for ans, (doc, _) in zip(answers, chunk):
            if ans == "S":
                uid = _unique_id(doc.metadata)
                if uid:
                    accepted.add(uid)
    return accepted

# ─── Conteo principal ───────────────────────────────────────────────
def _semantic_count(concept: str) -> int:
    ids_total: Set[str] = set()
    grey_all: list      = []
    with ThreadPoolExecutor() as exe:
        for ids_ok, grey in exe.map(_ids_for_term, _expand_terms(concept)):
            ids_total |= ids_ok
            grey_all  += grey
    ids_total |= _verify_docs(concept, grey_all)
    return len(ids_total)

# ─── Respuesta final ────────────────────────────────────────────────
def _format_answer(question: str, n: int, concept: str) -> str:
    return f"En la base hay **{n}** sentencias que mencionan {concept}."

def run(question: str) -> str:
    concept = _extract_concept(question)
    total   = _semantic_count(concept)
    return _format_answer(question, total, concept)

# ─── CLI de prueba ──────────────────────────────────────────────────
if __name__ == "__main__":
    try:
        while True:
            q = input("\n🗨️  Pregunta: ")
            print(run(q))
    except KeyboardInterrupt:
        pass
