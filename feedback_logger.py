"""
feedback_logger.py
------------------
Registra cada interacción para RLHF continuo.

Formato Excel (`Interactions.xlsx`) con columnas:
    • timestamp      (UTC ISO)
    • user_msg       (texto crudo)
    • assistant_msg  (respuesta Markdown)
    • intent         (etiqueta del router)
    • feedback       (Acepta/Parcial/Rechaza)

Este archivo se lee luego para entrenar el modelo de recompensa
o exportar a JSONL para DPO / RLHF fine-tuning:contentReference[oaicite:0]{index=0}.
"""

import pandas as pd
from datetime import datetime, timezone
from pathlib import Path
from typing import Literal
from config import INTER_FILE, SCORES

def _ensure_file() -> None:
    if not Path(INTER_FILE).exists():
        df = pd.DataFrame(columns=[
            "timestamp", "user_msg", "assistant_msg", "intent", "feedback"
        ])
        df.to_excel(INTER_FILE, index=False)

def log_interaction(
    user_msg: str,
    assistant_msg: str,
    intent: str,
    feedback: Literal["Acepta", "Parcial", "Rechaza"]
) -> None:
    _ensure_file()
    df = pd.read_excel(INTER_FILE)
    df.loc[len(df)] = [
        datetime.now(timezone.utc).isoformat(),
        user_msg,
        assistant_msg,
        intent,
        feedback,
    ]
    df.to_excel(INTER_FILE, index=False)
