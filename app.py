# app.py (refactor para evitar duplicación y usar st.chat_message)

import streamlit as st
from agent import responder_pregunta
from feedback_logger import log_interaction
from router import detect_intent

st.set_page_config(page_title="Asistente Judicial • RAG + Llama-4")

if "history" not in st.session_state:
    st.session_state.history = []

st.title("Asistente Judicial · RAG + Llama-4 Maverick")

# Mostrar historial completo con burbujas de chat
for q, a in st.session_state.history:
    with st.chat_message("user"):
        st.markdown(q)
    with st.chat_message("assistant"):
        st.markdown(a)

# Entrada del usuario
query = st.chat_input("Escribe tu consulta judicial…")

if query:
    # Mostrar mensaje del usuario
    with st.chat_message("user"):
        st.markdown(query)

    # Obtener respuesta
    answer = responder_pregunta(query)

    # Mostrar respuesta con botones de feedback
    with st.chat_message("assistant"):
        st.markdown(answer)
        col1, col2, col3 = st.columns(3)
        with col1:
            if st.button("👍 Acepta", key=f"acepta_{len(st.session_state.history)}"):
                log_interaction(query, answer, detect_intent(query), "Acepta")
        with col2:
            if st.button("🤷 Parcial", key=f"parcial_{len(st.session_state.history)}"):
                log_interaction(query, answer, detect_intent(query), "Parcial")
        with col3:
            if st.button("👎 Rechaza", key=f"rechaza_{len(st.session_state.history)}"):
                log_interaction(query, answer, detect_intent(query), "Rechaza")

    # Añadir a historial
    st.session_state.history.append((query, answer))
