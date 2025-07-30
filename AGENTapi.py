# =============================
# Sentencia QA – FastAPI Backend (v1.2)
# =============================
# Se eliminó por completo la dependencia de **Streamlit** en todos los
# módulos cargados por Uvicorn (router.py, agent.py).  La API sigue
# exponiendo un único endpoint POST /qa que acepta preguntas en lenguaje
# natural; el identificador se detecta automáticamente.
#
# ┌────────────────────────────────────────────────────────┐
# │ 1. app/main.py   ← entry‑point FastAPI                │
# │ 2. core/agent.py ← orquestador de intents y tools     │
# │ 3. core/router.py ← clasificador vía LLM              │
# └────────────────────────────────────────────────────────┘
#
# =========================================================
# 1. File: app/main.py
# =========================================================
from fastapi import FastAPI, HTTPException, status
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware



app = FastAPI(
    title="Sentencia QA API",
    version="1.2.0",
    docs_url="/docs",
    redoc_url="/redoc",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class QARequest(BaseModel):
    question: str

class QAResponse(BaseModel):
    answer: str

@app.get("/health", tags=["health"])
async def health():
    return {"status": "ok"}

@app.post("/qa", response_model=QAResponse, tags=["qa"])
async def qa_endpoint(req: QARequest):
    answer = "No fue posible responder la pregunta. trate mas tarde..."
    try:    
        answer = responder_pregunta(req.question)
        if answer.startswith("⚠️"):
            raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=answer)
    except Exception as e:
        logging.error(e)
    return {"answer": answer}

# =========================================================
# 2. File: core/agent.py
# =========================================================
import logging
from memory import memory
from tools import consulta_doc
from tools.expediente import run as expediente_run
from tools.resumen_doc import run as resumen_run
from tools.estadistica import run as estadistica_run
from tools.comparar import run as comparar_run
from tools.borrador_alerta import run as alerta_run
from tools.consulta_doc import run as consulta_run, _set_active, extract_identifier
#from tools.auditoria_ley import run as auditoria_ley_run
from tools.query_libre import query_libre_run
from vectorstore import search_by_text
from tools.estadistica_ai import run as estadistica_ai_run
from tools.query_libre import query_libre_run
from tools.cronología import run as cronologia_run
logger = logging.getLogger(__name__)

TOOL_MAP = {
    "expediente": expediente_run,
    "resumen_doc": resumen_run,
    "estadistica": estadistica_run,
    "estadistica_ai": estadistica_ai_run,
    "comparar_juris": comparar_run,
    "borrador_alerta": alerta_run,
    "consulta_doc": consulta_run, # ← NUEVO label
    "consulta_concepto": query_libre_run,
    "cronologia": cronologia_run,
    #"auditoria_ley": auditoria_ley_run
}

def auto_activate_if_id(text: str):
    ident = extract_identifier(text)
    if not ident:
        return
    hit = search_by_text(ident, k=1)
    if hit:
        _set_active(hit[0])

def responder_pregunta(msg: str) -> str:
    respuesta = ""
    try:
        auto_activate_if_id(msg)
        label = detect_intent(msg)
        print(label)
        logger.debug("Intento clasificado: %s", label)
        if label == "desconocido" and resumen_tool._get_pending():
            label = "resumen_doc"
        if label == "desconocido":
            if consulta_doc.active_doc:
                return consulta_doc.run(msg)
            return (
                "⚠️ No estoy seguro de si tu petición es de naturaleza judicial. "
                "¿Podrías darme más detalles o reformular la pregunta?"
            )

        respuesta = TOOL_MAP[label](msg)
        print(respuesta, label)
        combined_input = f"[Intent: {label}] {msg}"
        memory.save_context({"user": combined_input}, {"assistant": respuesta})
    except Exception as e:
        logger.exception("Error al procesar la pregunta")
        respuesta = (
            "⚠️ No estoy seguro de si tu petición es de naturaleza judicial. "
            "¿Podrías darme más detalles o reformular la pregunta?"
        )
    return respuesta

# =========================================================
# 3. File: core/router.py
# =========================================================
import logging
from together import Together
from typing import Literal
from config import TOGETHER_API_KEY, LLM_MODEL_ID
from sentence_transformers import SentenceTransformer
from router import *
logger = logging.getLogger(__name__)

LABELS = (
    "expediente",
    "resumen_doc",
    "consulta_doc",   # ← NUEVO label
    "estadistica",
    "estadistica_ai",
    "comparar_juris",
    "consulta_concepto",
    "borrador_alerta",
    "cronologia",
    #"auditoria_ley",
    "desconocido",
)

client = Together(api_key=TOGETHER_API_KEY)

