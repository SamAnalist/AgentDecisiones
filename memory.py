# memory.py
# ----------
# Manejo de memoria por sesión:
# 1) ConversationBufferMemory (últimos k turnos)
# 2) Estado de sesión (docs, doc_actual, pdf_cache, etc.)

from typing import Dict, Optional, List
from langchain.memory import ConversationBufferMemory

# ------------- 1. Conversación (k últimos turnos) -----------------
memory = ConversationBufferMemory(k=12, return_messages=True)

# Un ConversationBufferMemory independiente por session_id
_conversation_memories: Dict[str, ConversationBufferMemory] = {}
# 👇 Añadir estas utilidades al final de memory.py (después de register_pdf)

def get_pdf_text(session_id: str, pid: str) -> str | None:
    """
    Recupera el texto completo de un PDF previamente registrado
    mediante register_pdf().  Devuelve None si no existe.
    """
    return get_session_state(session_id).get("pdf_cache", {}).get(pid)


def get_conversation_memory(session_id: str) -> ConversationBufferMemory:
    if session_id not in _conversation_memories:
        _conversation_memories[session_id] = ConversationBufferMemory(
            k=12, return_messages=True
        )
    return _conversation_memories[session_id]


# ------------- 2. Estado de sesión extendido ----------------------
_sessions: Dict[str, dict] = {}


def get_session_state(session_id: str) -> dict:
    """Devuelve o inicializa el dict de estado para la sesión."""
    return _sessions.setdefault(session_id, {})


# ──────────────────── API legacy: active_doc ──────────────────────
def set_active_doc(session_id: str, doc_id: str):
    get_session_state(session_id)["active_doc"] = doc_id


def get_active_doc(session_id: str) -> Optional[str]:
    return get_session_state(session_id).get("active_doc")


# ───────────── NUEVO: gestión de múltiples PDFs por sesión ────────
def get_docs(session_id: str) -> List[dict]:
    """
    Lista de documentos cargados.
    Cada entrada: {
        'pid': str,          # hash interno: pdf@xxxx
        'name': str,         # nombre original del archivo
        'text': str,         # texto completo extraído
        'embedding': list | None,  # vector opcional (dim d)
    }
    """
    return get_session_state(session_id).setdefault("docs", [])


def set_doc_actual(session_id: str, pid: str) -> None:
    """Establece el PDF en foco (lo que el usuario está consultando)."""
    get_session_state(session_id)["doc_actual"] = pid


def get_doc_actual(session_id: str) -> Optional[str]:
    return get_session_state(session_id).get("doc_actual")


def register_pdf(
    session_id: str,
    pid: str,
    texto: str,
    *,
    name: str = "",
    embedding: List[float] | None = None,
) -> None:
    """
    Registra un PDF en la sesión:
    • Guarda el texto en pdf_cache[pid]
    • Añade/actualiza entrada en docs (pid, name, text, embedding)
    • Marca doc_actual = pid
    """
    state = get_session_state(session_id)

    # 1) cache rápido por pid (retro-compatibilidad)
    state.setdefault("pdf_cache", {})[pid] = texto

    # 2) docs list
    docs = state.setdefault("docs", [])
    for d in docs:
        if d["pid"] == pid:
            # Ya existe → actualiza name/embedding si están vacíos
            if name and not d["name"]:
                d["name"] = name
            if embedding is not None and d.get("embedding") is None:
                d["embedding"] = embedding
            break
    else:
        # Nuevo documento
        docs.append(
            {
                "pid": pid,
                "name": name or pid,
                "text": texto,
                "embedding": embedding,
            }
        )

    # 3) foco actual
    state["doc_actual"] = pid
# ---------------- pending action (clarificación) -----------------
def set_pending(session_id: str, data: dict | None):
    """
    Guarda la acción pendiente (ej.: {'intent':'resumen_doc','need':1})
    Si data es None, borra la acción pendiente.
    """
    st = get_session_state(session_id)
    if data is None:
        st.pop("pending", None)
    else:
        st["pending"] = data


def get_pending(session_id: str) -> dict | None:
    """Devuelve la acción pendiente o None."""
    return get_session_state(session_id).get("pending")
