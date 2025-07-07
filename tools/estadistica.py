"""
tools.estadistica
-----------------
Consultas tipo “¿cuántos casos…?” sobre el DataFrame de metadatos.

• Usa ChatTogether vía langchain-experimental.
• Limpia nombres de columnas y rellena las que falten, de modo que
  no reviente si el Excel tiene encabezados distintos.
"""

import pandas as pd
from langchain_experimental.agents import create_pandas_dataframe_agent
from langchain_together import ChatTogether
from config import DATA_DIR, TOGETHER_API_KEY, LLM_MODEL_ID

# ── 1. Cargar Excel completo y normalizar encabezados ─────────────
df_meta = pd.read_excel(DATA_DIR / "output (1).xlsx")

# Quitar espacios y uniformar mayúsculas
df_meta.columns = [c.strip() for c in df_meta.columns]

# Mapas de alias posibles → nombre estándar
ALIAS = {
    "nuc": "NUC",
    "número de caso": "NUC",
    "id_documento": "IdDocumento",
    "iddocumento": "IdDocumento",
}

df_meta.rename(columns={k: v for k, v in ALIAS.items() if k in df_meta.columns}, inplace=True)

# Asegurar columnas requeridas
NEEDED = ["IdDocumento", "NUC", "Materia", "Asunto", "TipoFallo"]
for col in NEEDED:
    if col not in df_meta.columns:
        df_meta[col] = ""

# Mantener solo las necesarias (otras no molestan, pero aclaramos)
df_meta = df_meta[NEEDED]

# ── 2. Crear agente de pandas (Together) ──────────────────────────
llm = ChatTogether(
    together_api_key=TOGETHER_API_KEY,
    model_name=LLM_MODEL_ID,
    temperature=0.0,
)

agent = create_pandas_dataframe_agent(
    llm,
    df_meta,
    verbose=False,
    allow_dangerous_code=True,   # ← confirmas que aceptas el REPL interno
)

def run(msg: str) -> str:
    """Devuelve resultado de la consulta o error legible."""
    try:
        answer = agent.run(msg)
        return f"**Resultado**\n{answer}"
    except Exception as e:
        return f"⚠️ No pude procesar la consulta estadística: {e}"
