# agent.py — Orquestador con confirmación y ejecución de herramientas

from together import Together
from memory import memory
from router import detect_intent
from tools import consulta_doc
from tools.expediente import run as expediente_run
from tools.resumen_doc import run as resumen_run
from tools.estadistica import run as estadistica_run
from tools.comparar import run as comparar_run
from tools.borrador_alerta import run as alerta_run
from tools.consulta_doc import run as consulta_run, _set_active, extract_identifier  # ← importa setter
from tools.estadistica_ai import run as estadistica_ai_run
TOOL_MAP = {
    "expediente": expediente_run,
    "resumen_doc": resumen_run,
    "estadistica": estadistica_run,
    "estadistica_ai": estadistica_ai_run,
    "comparar_juris": comparar_run,
    "borrador_alerta": alerta_run,
    "consulta_doc": consulta_run # ← NUEVO label
}
from tools.consulta_doc import extract_identifier, _set_active
from vectorstore import search_by_text
def auto_activate_if_id(text: str):
    ident = extract_identifier(text)
    if not ident:
        return
    hit = search_by_text(ident, k=1)
    if hit:
        _set_active(hit[0])
def responder_pregunta(msg: str) -> str:
    auto_activate_if_id(msg)   # <── SIEMPRE ejecuta esto primero
    # Detecta la intención mediante el router
    label = detect_intent(msg)
    import streamlit as st
    st.write(label)
    if label == "desconocido" and consulta_doc.active_doc:
        # trata de contestar usando el documento activo
        return consulta_doc.run(msg)
    elif label == "desconocido":
        return (
            "⚠️ No estoy seguro de si tu petición es de naturaleza judicial. "
            "¿Podrías darme más detalles o reformular la pregunta?"
        )


        # Ejecuta la herramienta correspondiente
    respuesta = TOOL_MAP[label](msg)

    # Guardar contexto en memoria — combinamos user y intent en una sola cadena
    combined_input = f"[Intent: {label}] {msg}"
    memory.save_context({"user": combined_input}, {"assistant": respuesta})

    return respuesta
