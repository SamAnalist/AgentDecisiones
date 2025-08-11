# app.py — frontend local Streamlit con soporte multi-PDF
import streamlit as st
import fitz        # PyMuPDF
import uuid

from agent import responder_pregunta       # acepta (session_id, question, files)
from feedback_logger import log_interaction
from router import detect_intent

st.set_page_config(page_title="Asistente Judicial")

# ───────────── estado de sesión ─────────────
if "history" not in st.session_state:
    st.session_state.history = []

if "session_id" not in st.session_state:
    st.session_state.session_id = uuid.uuid4().hex

session_id = st.session_state.session_id

st.title("Asistente Judicial · Poder Judicial")

# ──────── 1. Subir uno o varios PDFs ──────────────────────
uploaded_files = st.file_uploader(
    "Adjunta uno o varios PDFs",
    type="pdf",
    accept_multiple_files=True,
)

files_payload = None
if uploaded_files:
    files_payload = []
    for up in uploaded_files:
        # Extraer texto con PyMuPDF
        with fitz.open(stream=up.getvalue(), filetype="pdf") as doc:
            text = "".join(p.get_text() for p in doc)
        files_payload.append({"name": up.name, "text": text})

    st.success(
        "He recibido " +
        ", ".join(f['name'] for f in files_payload) +
        ". Ahora puedes pedir resúmenes, expedientes o comparaciones."
    )

# ──────── 2. Mostrar historial ──────────────
for q, a in st.session_state.history:
    with st.chat_message("user"):
        st.markdown(q)
    with st.chat_message("assistant"):
        st.markdown(a)

# ──────── 3. Entrada de texto ───────────────
query = st.chat_input("Escribe tu consulta judicial…")

if query:
    with st.chat_message("user"):
        st.markdown(query)

    # Llamar al agente con session_id y lista de archivos (o None)
    answer = responder_pregunta(
        question=query
    )

    # Mostrar respuesta + feedback
    with st.chat_message("assistant"):
        st.markdown(answer)
        c1, c2, c3 = st.columns(3)
        if c1.button("👍 Acepta", key=f"ok_{len(st.session_state.history)}"):
            log_interaction(query, answer, detect_intent(query), "Acepta")
        if c2.button("🤷 Parcial", key=f"mid_{len(st.session_state.history)}"):
            log_interaction(query, answer, detect_intent(query), "Parcial")
        if c3.button("👎 Rechaza", key=f"bad_{len(st.session_state.history)}"):
            log_interaction(query, answer, detect_intent(query), "Rechaza")

    st.session_state.history.append((query, answer))
