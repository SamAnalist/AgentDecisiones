"""
Conteo semántico de expedientes (v2)
------------------------------------
• Extrae el concepto jurídico usando heurística + few-shot prompt
• Genera sinónimos afines
• Búsqueda híbrida FAISS + filtro keyword
• Zona gris: Cross-Encoder si está en cache; LLM S/N si no
"""
import sys, os, pathlib, re, unicodedata, numpy as np
from concurrent.futures import ThreadPoolExecutor
from functools import lru_cache
from typing import List, Set, Tuple
ROOT = pathlib.Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path: sys.path.insert(0, str(ROOT))

from config import TOGETHER_API_KEY, LLM_MODEL_ID, SIM_THRESHOLD_est, GREY_MARGIN
from embed import BNEEmbeddings
from tools.semantic_search import search_with_scores
from together import Together
from sentence_transformers import CrossEncoder
from sentence_transformers.util import snapshot_download

# ─── Modelos ─────────────────────────────────────────────────────────
embedder = BNEEmbeddings()
llm      = Together(api_key=TOGETHER_API_KEY)
try:
    snapshot_download("cross-encoder/ms-marco-MiniLM-L-6-v2", local_files_only=False)
    reranker = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2", device="cpu")
except Exception:
    reranker = None

# ─── Heurística de stop-words y normalización ───────────────────────
LEX_STOP = {"cuantos", "cuántos", "cuantas", "cuántas", "numero", "número",
            "tasa", "estadistica", "estadísticas", "porcentaje", "%",
            "promedio", "casos", "caso"}
def _strip_acc(s):  # sin tildes
    return ''.join(c for c in unicodedata.normalize("NFKD", s)
                   if unicodedata.category(c) != "Mn")

def _keyword_match(term: str, text: str) -> bool:
    base = re.sub(r"s$", "", _strip_acc(term.lower()))
    return re.search(rf"\b{re.escape(base)}\w*\b", _strip_acc(text.lower())) is not None

def _unique_id(m: dict | None) -> str | None:
    if not m: return None
    for k in ("DocumentID", "IdDocumento", "document_id",
              "NUC", "NumeroTramite", "numero_tramite"):   # añade aquí si falta
        if m.get(k): return str(m[k]).lower()

# ─── Prompt few-shot para concepto ──────────────────────────────────
FEW_SHOT = (
    "P: ¿Cuántos casos de robo hay?\nC: robo\n\n"
    "P: Promedio de indemnizaciones en homicidio culposo\nC: homicidio culposo\n\n"
    "P: % de demandas por divorcio en 2023\nC: divorcio\n\n"
)
def _extract_concept(q: str) -> str:
    tokens = [t for t in q.lower().split() if _strip_acc(t) not in LEX_STOP]
    if tokens and tokens[0] in {"de", "del"}: tokens = tokens[1:]
    seed = " ".join(tokens[:4]) or q
    prompt = (FEW_SHOT +
              f"P: {q.strip()}\nC:")
    rsp = llm.chat.completions.create(
        model=LLM_MODEL_ID,
        messages=[{"role": "user", "content": prompt}],
        temperature=0.0, max_tokens=12)
    concept = rsp.choices[0].message.content.strip().split("\n")[0]
    return concept.lower()

@lru_cache(maxsize=256)
def _expand_terms(base: str) -> List[str]:
    p = (f"Da hasta 5 sinónimos o delitos afines de «{base}» "
         f"(minúsculas, coma, sin explicación)")
    rsp = llm.chat.completions.create(
        model=LLM_MODEL_ID,
        messages=[{"role": "user", "content": p}],
        temperature=0.0, max_tokens=20)
    syns = [s.strip().lower() for s in rsp.choices[0].message.content.split(",")]
    return list({base, *syns})

# ─── Búsqueda híbrida ───────────────────────────────────────────────
def _ids_for_term(term: str, k: int = 800) -> Tuple[Set[str], list]:
    vec = embedder.embed_query(term)
    hits = search_with_scores(vec, k=k)
    ids, grey = set(), []
    lo, hi = SIM_THRESHOLD_est, SIM_THRESHOLD_est + GREY_MARGIN
    for d, dist in hits:
        if not _keyword_match(term, d.page_content): continue
        uid = _unique_id(d.metadata);  # ignora sin ID
        if not uid: continue
        if dist <= lo: ids.add(uid)
        if lo - GREY_MARGIN < dist <= hi: grey.append((d, dist))
    # re-rank
    if grey and reranker:
        pairs = [(term, d.page_content[:512]) for d, _ in grey]
        for (d, _), s in zip(grey, reranker.predict(pairs, batch_size=32)):
            if s > 0.8:
                uid = _unique_id(d.metadata)
                if uid: ids.add(uid)
        grey = []
    return ids, grey

def _verify(concept: str, grey: list) -> Set[str]:
    if not grey: return set()
    bullets = [f"{i+1}) {d.page_content[:240].replace(chr(10),' ')}"
               for i, (d, _) in enumerate(grey)]
    p = f"Concepto: «{concept}». Indica S/N si cada fragmento se relaciona.\n" \
        + "\n".join(bullets)
    ans = llm.chat.completions.create(
        model=LLM_MODEL_ID,
        messages=[{"role": "user", "content": p}],
        temperature=0.0, max_tokens=len(grey)*2
    ).choices[0].message.content.upper()
    acc = { _unique_id(d.metadata)
            for a,(d,_) in zip(re.findall(r"[SN]", ans), grey) if a=="S"}
    return {i for i in acc if i}

def _semantic_count(conc: str) -> int:
    ids, grey_all = set(), []
    with ThreadPoolExecutor() as ex:
        for ok, grey in ex.map(_ids_for_term, _expand_terms(conc)):
            ids |= ok; grey_all += grey
    ids |= _verify(conc, grey_all)
    return len(ids)

def _fmt(q, n, c): return f"En la base hay **{n}** sentencias que mencionan {c}."

def run(question: str) -> str:
    concept = _extract_concept(question)
    total = _semantic_count(concept)
    return _fmt(question, total, concept)

if __name__ == "__main__":
    try:
        while True: print(run(input("\n🗨️ ")))
    except KeyboardInterrupt: pass
