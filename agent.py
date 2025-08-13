# core/agent.py — Orquestador con soporte doc_text + session_id
import logging, uuid
from memory import memory, register_pdf
from router import detect_intent
from tools import consulta_doc
from tools.comparar_ids import run as comparar_ids_run
from tools.cronología import run as cronologia_run

# Tools adaptadas a (session_id, msg)
from tools.expediente     import run as expediente_run
from tools.resumen_doc    import run as resumen_run
from tools.comparar       import run as comparar_run

# Tools que NO necesitan session_id aún
from tools.estadistica    import run as estadistica_run
from tools.borrador_alerta import run as alerta_run
from tools.auditoria_ley  import run as auditoria_ley_run
from tools.query_libre    import query_libre_run

from tools.consulta_doc   import run as consulta_run, _set_active, extract_identifier
from vectorstore          import search_by_text
from tools.conversacional import run as conversacional_run

logger = logging.getLogger(__name__)

# Mapa, distinguiendo firmas
TOOL_MAP = {
    "expediente":      expediente_run,      # (session_id, msg)
    "resumen_doc":     resumen_run,         # (session_id, msg)
    "relacionar_juris":  comparar_run,        # (session_id, msg)
    "estadistica":     estadistica_run,     # (msg)
    #"borrador_alerta": alerta_run,          # (msg)
    "consulta_doc":    consulta_run,        # (msg) — ya usa consulta_doc.active_doc
    "consulta_concepto": query_libre_run,   # (msg)
    "auditoria_ley":   auditoria_ley_run,
    "comparar_ids":   comparar_ids_run,
    "conversacional": conversacional_run,
    "cronologia": cronologia_run
}

from typing import Tuple, Any, Optional

def ultima_interaccion_filtrada(conversacion: Any, n_chars: int = 100
                                ) -> Optional[Tuple[str, str]]:
    """
    Devuelve el último par (humano, bot[:n_chars]) ignorando los mensajes
    humanos cuyo intent sea 'comparar_ids' o 'relacionar_juris'.

    Parameters
    ----------
    conversacion : dict | dict_items
        Estructura con la clave 'history' (lista de HumanMessage / AIMessage).
    n_chars : int
        Número de caracteres a recortar de la respuesta del bot.

    Returns
    -------
    Tuple[str, str] | None
        (mensaje_humano, preview_bot) o None si no se encontró un par válido.
    """
    data = dict(conversacion) if not isinstance(conversacion, dict) else conversacion
    history = data.get("history", [])

    # Recorremos de atrás hacia delante para encontrar la última interacción válida
    i = len(history) - 1
    while i >= 0:
        # Queremos que history[i] sea HumanMessage y history[i+1] el AIMessage posterior
        if isinstance(history[i], type(history[0])) and "content" in history[i].__dict__:
            content = history[i].content
            # Verificamos que no contenga los intents prohibidos
            if not any(intent in content for intent in ("[Intent: comparar_ids",
                                                        "[Intent: relacionar_juris",
                                                        "[Intent: conversacional")):
                # Aseguramos que haya una respuesta de bot inmediatamente después
                if i + 1 < len(history):
                    ai_msg = history[i + 1]
                    ai_preview = getattr(ai_msg, "content", "")[:n_chars]
                    return (content, ai_preview)
        i -= 1

    # Si no encontramos nada que cumpla las condiciones
    return None


# ───────────────────────── helpers ─────────────────────────
def _auto_activate_if_id(text: str):
    ident = extract_identifier(text)
    if not ident:
        return
    hit = search_by_text(ident, k=1)
    if hit:
        _set_active(hit[0])
history = ultima_interaccion_filtrada(memory.load_memory_variables({}))

# ───────────────────────── función pública ─────────────────
def responder_pregunta(
    question: str,
) -> str:
    global history

    """
    Orquesta la respuesta:
    • si llega `doc_text` lo registra y marca como documento activo.
    • detecta intención → llama herramienta adecuada.
    """

    # 3) Detectar intención con el router
    label = detect_intent(question)
    logger.debug("Intento clasificado: %s", label)

    # if label == "conversacional":
    #     if consulta_doc.active_doc:
    #         return consulta_doc.run(question)
    #     try:
    #         conversacional_run(question)
    #     except:
    #         return (
    #         "⚠️ No estoy seguro de si tu petición es de naturaleza judicial. "
    #         "¿Podrías darme más detalles o reformular la pregunta?"
    #     )
    use_hist = (label == "conversacional")
    msg = question if not use_hist else f"{question}\n\nHistorial de conversaciones:\n{history}"
    # 3) Llama a la tool
    print(msg)
    respuesta = TOOL_MAP[label](msg)
    # 5) Guardar en memoria de conversación
    memory.save_context(
        {"user": f"[Intent: {label}] {question}"},
        {"assistant": respuesta},
    )
    history = memory.load_memory_variables({})
    return respuesta
