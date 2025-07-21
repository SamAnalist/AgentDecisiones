# tools/consulta_doc.py  — QA multi-llamada sobre la sentencia ACTIVA
"""
• Usa exactamente los mismos chunks que resumen_doc (cargados desde /chunks/*.json)
• Si el contexto supera MAX_CHARS_PER_CALL, lo divide en varios requests

Este módulo se ha simplificado para **eliminar cualquier dependencia de Streamlit**.  Ahora la
única memoria de trabajo es la variable de módulo `active_doc`.  Si necesitas
persistencia entre peticiones, gestiona el identificador de sentencia a nivel de
llamada (por ejemplo, almacenándolo en la base de datos o en un token de
sesión) y pásalo explícitamente a tu lógica.
"""

from __future__ import annotations
import os, json, re
from typing import Optional

from together import Together
from config import TOGETHER_API_KEY, LLM_MODEL_ID, DATA_DIR
from memory import memory

# ───────────────────────── LLM ──────────────────────────
_client = Together(api_key=TOGETHER_API_KEY)

# ─────────────── Estado en memoria ─────────────────────
# El documento activo se mantiene sólo en memoria de proceso.
active_doc: Optional[dict] = None

# ────────────── CARGA DE CHUNKS (idéntico a resumen_doc) ──────────────
CHUNKS_DIR = os.path.join(DATA_DIR, "chunks")
docs_map: dict[str, list[str]] = {}

for fname in os.listdir(CHUNKS_DIR):
    if not fname.endswith(".json"):
        continue
    with open(os.path.join(CHUNKS_DIR, fname), "r", encoding="utf-8") as f:
        entry = json.load(f)

    md   = entry.get("metadata", {})
    text = entry.get("text", "")

    keys = [
        str(md.get("NUC", "")),
        str(md.get("NumeroTramite", "")),
        str(md.get("DocumentID", "")),
    ]
    for k in keys:
        k_norm = k.lower().lstrip("auto:").strip()
        if k_norm:
            docs_map.setdefault(k_norm, []).append(text)

# ────────────── ID helpers ────────────────────────────────────────────
ID_PATTERN = re.compile(
    r"""(
        \b\d{3}-\d{4}-[A-Z]+-\d{5}\b      # 034-2020-ECON-00096
      | \b\d{3}-\d{4}-[A-Z]+-\d{3,}\b     # 533-2020-ECON-00751
      | SCJ[-/]\w+[-/]\d{4}[-/]\d+        # SCJ-TS-2023-1234
      | \b\d{4}-\d{1,5}\b                 # 2023-123
    )""",
    re.VERBOSE | re.IGNORECASE,
)

def extract_identifier(text: str) -> Optional[str]:
    m = ID_PATTERN.search(text)
    return m.group(1) if m else None

def _doc_id(doc) -> Optional[str]:
    # 1) Si viene como dict plano
    if isinstance(doc, dict):
        for key in ("NUC", "NumeroTramite", "DocumentID"):
            if doc.get(key):
                return str(doc[key]).lower().lstrip("auto:").strip()
    # 2) Si es un objeto con .metadata
    meta = getattr(doc, "metadata", {}) or {}
    for key in ("NUC", "NumeroTramite", "DocumentID", "doc_id", "case_number"):
        if meta.get(key):
            return str(meta[key]).lower().lstrip("auto:").strip()
    return None


def _set_active(doc):
    """Define el documento activo en memoria."""
    global active_doc
    active_doc = doc


# ────────────── QA multi-llamada (divide si es demasiado largo) ───────
MAX_CHARS_PER_CALL = 100_000             # ≈ 25 000 tokens

def _split_by_size(text: str, max_chars: int):
    pos = 0
    while pos < len(text):
        end = min(pos + max_chars, len(text))
        cut = text.rfind("\n", pos, end) or text.rfind(" ", pos, end)
        if cut <= pos:
            cut = end
        yield text[pos:cut].strip()
        pos = cut

def _ask_llm(system: str, user: str, max_tok: int = 768) -> str:
    resp = _client.chat.completions.create(
        model=LLM_MODEL_ID,
        messages=[{"role": "system", "content": system},
                  {"role": "user",   "content": user}],
        temperature=0.0,
        max_tokens=max_tok,
    )
    return resp.choices[0].message.content.strip()

def _qa_part(context: str, question: str) -> str:
    sys = (
        "Eres un asistente jurídico experto en jurisprudencia dominicana. "
        "Responde SOLO con la información en CONTEXTO. "
        "Si falta, di: 'No hay información suficiente en la sentencia'."
    )
    usr = f"CONTEXTO:\n{context}\n\nPREGUNTA:\n{question}"
    return _ask_llm(sys, usr)

def _qa_multi(context_full: str, question: str) -> str:
    if len(context_full) <= MAX_CHARS_PER_CALL:
        return _qa_part(context_full, question)

    answers: list[str] = []
    for chunk in _split_by_size(context_full, MAX_CHARS_PER_CALL):
        answers.append(_qa_part(chunk, question))

    merged = "\n".join(dict.fromkeys(
        line for ans in answers for line in ans.splitlines() if line.strip()
    ))
    return merged or "No hay información suficiente en la sentencia."

# ────────────── build context (TODOS los chunks) ──────────────────────

def _build_context(doc_id_norm: str) -> str:
    chunks = docs_map.get(doc_id_norm, [])
    return "\n\n".join(chunks)

# ────────────── API principal ─────────────────────────────────────────

def run(user_msg: str) -> str:
    """Ejecuta la consulta sobre la sentencia activa o la que indique el usuario."""
    global active_doc

    ident = extract_identifier(user_msg)
    if ident:
        ident_norm = ident.lower().strip()
        _set_active({
            "NUC": ident_norm,
            "NumeroTramite": ident_norm,
            "DocumentID": ident_norm,
        })

    if not active_doc:
        return "⚠️ No hay ninguna sentencia activa. Indica el número de caso."

    did = _doc_id(active_doc)
    if not did:
        return "⚠️ La sentencia activa no tiene identificador reconocible."

    context = _build_context(did)
    if not context:
        return "⚠️ Texto de la sentencia no encontrado en memoria."

    answer = _qa_multi(context, user_msg)
    memory.save_context({"user": user_msg}, {"assistant": answer})
    return answer
