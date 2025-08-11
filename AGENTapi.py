# AGENTapi.py  (o app/main.py)

from fastapi import FastAPI, HTTPException, Header, status
from pydantic import BaseModel
from typing import List, Optional
import uuid

from agent import responder_pregunta   # apunta al nuevo agent.py

app = FastAPI(title="Sentencia QA API", version="1.3")

# ---------- modelos de request/response --------------------------
class FileItem(BaseModel):
    name: str
    text: str


class QARequest(BaseModel):
    question: str = ""
    archivos: Optional[List[FileItem]] = None   # ← múltiple PDFs


class QAResponse(BaseModel):
    answer: str


# -------------------- endpoint /qa -------------------------------
@app.post("/qa", response_model=QAResponse, tags=["qa"])
async def qa_endpoint(
    req: QARequest,
    x_session: Optional[str] = Header(None, convert_underscores=False),
):
    # 1) sesión
    session_id = x_session or uuid.uuid4().hex

    # 2) llamar al agente
    answer = responder_pregunta(
        question=req.question,
    )

    # 3) errores semánticos
    if answer.startswith("⚠️"):
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=answer)

    # 4) devolver session_id si era nuevo
    headers = {"X-Session": session_id} if not x_session else {}
    return QAResponse(answer=answer), headers
