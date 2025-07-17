"""
CONFIGURACIÓN GLOBAL para Juriteca AI (versión Together-only).
"""

from pathlib import Path

# ──────────── Rutas ────────────
BASE_DIR   = Path(__file__).parent
DATA_DIR   = BASE_DIR / "data"
INDEX_DIR  = BASE_DIR / "index_dir"
DATA_DIR.mkdir(exist_ok=True)
INDEX_DIR.mkdir(exist_ok=True)

# ──────────── Together AI ────────────
#  👉  export TOGETHER_API_KEY="tu-api-key"
import os
TOGETHER_API_KEY  = os.getenv("TOGETHER_API_KEY", "ff5f86ace267edc120015454ba49a48270e306ee3fbfe67952c5c8630655283d")
TOGETHER_BASE_URL = "https://api.together.xyz/v1"

# Modelo LLM (chat)
LLM_MODEL_ID = "meta-llama/Llama-4-Maverick-17B-128E-Instruct-FP8"

# Modelo de embeddings público en Together AI
EMBED_MODEL_ID = (
    "dariolopez/roberta-base-bne-finetuned-msmarco-qa-es-mnrl-mn"
)
# ──────────── VectorStore ────────────
CHUNK_SIZE      = 750       # tokens aprox.
CHUNK_OVERLAP   = 80
SIM_THRESHOLD   = 0.5     # coseno mínimo para “encontrado”
K_RETRIEVE      = 5
SIM_THRESHOLD_est = 1.0            # <= 1.0 se considera match
GREY_MARGIN   = 0.15

# ──────────── Feedback ────────────
SCORES     = {"Acepta": 1, "Parcial": 0, "Rechaza": -1}
INTER_FILE = DATA_DIR / "Interactions.xlsx"
