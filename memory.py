# memory.py
# ---------
# Manejo de memoria por sesión:
# 1) Buffer de conversación (últimos k turnos) por session_id
# 2) Estado de sesión (active_doc) por session_id

from typing import Dict, Optional
from langchain.memory import ConversationBufferMemory

memory = ConversationBufferMemory(
    k=12,
    return_messages=True,
)
# ───────────── Conversation memories ────────────────
# Un ConversationBufferMemory independiente por session_id
_conversation_memories: Dict[str, ConversationBufferMemory] = {}

def get_conversation_memory(session_id: str) -> ConversationBufferMemory:
    """
    Devuelve (o crea) el ConversationBufferMemory de la sesión dada.
    Este buffer guarda los últimos k mensajes (user ↔ assistant).
    """
    if session_id not in _conversation_memories:
        # k=12 como en tu configuración original
        _conversation_memories[session_id] = ConversationBufferMemory(
            k=12,
            return_messages=True
        )
    return _conversation_memories[session_id]


# ───────────── Session state ────────────────────────
# Guarda cualquier variable de estado extra: active_doc, flags, etc.
_sessions: Dict[str, dict] = {}

def get_session_state(session_id: str) -> dict:
    """
    Devuelve o inicializa el diccionario de estado para esta sesión.
    Aquí guardamos active_doc y cualquier otra variable por sesión.
    """
    return _sessions.setdefault(session_id, {})

def set_active_doc(session_id: str, doc_id: str):
    """
    Marca el documento activo (por su ID normalizado) en el estado de la sesión.
    """
    state = get_session_state(session_id)
    state["active_doc"] = doc_id

def get_active_doc(session_id: str) -> Optional[str]:
    """
    Recupera el documento activo de la sesión, o None si no hay ninguno.
    """
    return get_session_state(session_id).get("active_doc")
