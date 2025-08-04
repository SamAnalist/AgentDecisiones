# tools/consulta_doc.py  — QA multi-llamada sobre la sentencia ACTIVA + LEYES
"""
• Sigue usando los mismos chunks de la sentencia que resumen_doc (DATA_DIR/chunks/*.json).
• Si el usuario pregunta algo y ya hay un documento activo, ahora también
  se consultan los artículos constitucionales y criterios jurisprudenciales
  embebidos en index_dir/index_laws.

Flujo:
1. Si el mensaje contiene un identificador (NUC, IdDocumento, etc.) se activa.
2. Se arma CONTEXTO_SENTENCIA con todos los chunks del caso activo.
3. Se buscan hasta 5 leyes/criterios relevantes al mensaje → CONTEXTO_LEYES.
4. Se pregunta al LLM con ambos contextos. El sistema debe priorizar la
   sentencia; sólo complementa con leyes/criterios si son pertinentes.
"""

from __future__ import annotations
import os, json, re
from typing import Optional

from together import Together
from config import TOGETHER_API_KEY, LLM_MODEL_ID, DATA_DIR
from vectorstore import law_search                       # ← NEW
from memory import memory as _memory                     # para save_context

# ───────────────────────── LLM ──────────────────────────
_client = Together(api_key=TOGETHER_API_KEY)

# ─────────────── Estado en memoria ─────────────────────
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

    for key in ("NUC", "NumeroTramite", "DocumentID"):
        k_norm = str(md.get(key, "")).lower().lstrip("auto:").strip()
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
    for key in ("NUC", "NumeroTramite", "DocumentID", "doc_id", "case_number"):
        if isinstance(doc, dict) and doc.get(key):
            return str(doc[key]).lower().lstrip("auto:").strip()
        if not isinstance(doc, dict):
            meta = getattr(doc, "metadata", {}) or {}
            if meta.get(key):
                return str(meta[key]).lower().lstrip("auto:").strip()
    return None

def _set_active(doc):
    global active_doc
    active_doc = doc

# ────────────── QA helpers ────────────────────────────────────────────
MAX_CHARS_PER_CALL = 100_000             # ≈ 25 000 tokens

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
        "Eres un asistente jurídico experto en jurisprudencia dominicana.\n"
        "Responde PRIORITARIAMENTE con la información de CONTEXTO_SENTENCIA.\n"
        "Únicamente complementa con CONTEXTO_LEYES si aporta artículos o "
        "criterios explícitos que respalden la respuesta.\n"
        "Si la información no está disponible, responde: "
        "'No hay información suficiente en la sentencia'."
    )
    usr = f"{context}\n\nPREGUNTA:\n{question}"
    return _ask_llm(sys, usr)

def _qa_multi(context_full: str, question: str) -> str:
    if len(context_full) <= MAX_CHARS_PER_CALL:
        return _qa_part(context_full, question)

    answers = []
    for chunk in _split_by_size(context_full, MAX_CHARS_PER_CALL):
        answers.append(_qa_part(chunk, question))

    merged = "\n".join(dict.fromkeys(
        line for ans in answers for line in ans.splitlines() if line.strip()
    ))
    return merged or "No hay información suficiente en la sentencia."

# ────────────── build contexts ────────────────────────────────────────
def _build_context_sentencia(doc_id_norm: str) -> str:
    return "\n\n".join(docs_map.get(doc_id_norm, []))

def _build_context_leyes(pregunta: str) -> str:
    hits = law_search(pregunta, k=5)
    if not hits:
        return ""
    bloques = []
    for d in hits:
        fuente = d.metadata.get("fuente")
        if fuente == "constitucion":
            titulo = f"[Art. {d.metadata.get('articulo')}]"
        else:
            titulo = f"[Criterio {d.metadata.get('ID')}]"
        bloques.append(f"{titulo} {d.page_content}…")
    return "\n\n".join(bloques)

# ─────────── API principal ─────────────────────────────
def run(user_msg: str, memory=None) -> str:
    global active_doc

    ident = extract_identifier(user_msg)
    if ident:
        ident_n = ident.lower().strip()
        _set_active({"NUC": ident_n, "NumeroTramite": ident_n, "DocumentID": ident_n})

    if not active_doc:
        return "⚠️ No hay ninguna sentencia activa. Indica el número de caso."

    did = _doc_id(active_doc)
    if not did:
        return "⚠️ La sentencia activa no tiene identificador reconocible."

    ctx_sent = _build_context_sentencia(did)
    if not ctx_sent:
        return "⚠️ Texto de la sentencia no encontrado en memoria."

    ctx_leyes = _build_context_leyes(user_msg)
    contexto_completo = (
        "===== CONTEXTO_SENTENCIA =====\n" + ctx_sent +
        "\n\n===== CONTEXTO_LEYES =====\n" + (ctx_leyes or "—")
    )

    answer = _qa_multi(contexto_completo, user_msg)

    if memory is not None:
        _memory.save_context({"user": user_msg}, {"assistant": answer})

    return answer
