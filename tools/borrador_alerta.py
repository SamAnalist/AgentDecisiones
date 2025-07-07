"""
tools/borrador_alerta.py
-------------------------
Genera alertas de plazos y un borrador de fallo.
Maneja respuestas JSON o texto libre sin romper la app.
"""

import re
import json
from datetime import datetime, timedelta
import pandas as pd
from together import Together
from config import DATA_DIR, TOGETHER_API_KEY, LLM_MODEL_ID

# ── Carga plazos si existen ────────────────────────────────────────
plazos_file = DATA_DIR / "Plazos.xlsx"
if plazos_file.exists():
    df_plazos = pd.read_excel(plazos_file)
else:
    df_plazos = pd.DataFrame(columns=["NUC", "Actuacion", "FechaVenc"])

# ── Cliente Together ───────────────────────────────────────────────
_client = Together(api_key=TOGETHER_API_KEY)

# ── Regex para capturar NUC ────────────────────────────────────────
_PAT_NUC = re.compile(r"\b\d{3}-\d{4}-[A-Z]{4}-\d{5}\b", re.I)

# ── Función interna de alertas ────────────────────────────────────
def _alertas(nuc: str) -> str:
    hoy = datetime.now()
    proximos = df_plazos[
        (df_plazos["NUC"] == nuc) &
        (df_plazos["FechaVenc"] - hoy < timedelta(days=10))
    ]
    if proximos.empty:
        return "Sin plazos próximos (<10 días)."
    return "\n".join(
        f"- {row.Actuacion} vence el {row.FechaVenc:%d-%m-%Y}"
        for _, row in proximos.iterrows()
    )

# ── Función pública ────────────────────────────────────────────────
def run(msg: str) -> str:
    # Extraer NUC si lo hay
    m = _PAT_NUC.search(msg)
    nuc = m.group() if m else None

    # Llamada al LLM pidiendo JSON
    prompt = (
        "Devuelve SOLO JSON con clave 'borrador' que contenga "
        "el texto del borrador de fallo para esta petición judicial:\n\n"
        f"{msg}"
    )
    resp = _client.chat.completions.create(
        model=LLM_MODEL_ID,
        messages=[{"role": "user", "content": prompt}],
        response_format={"type": "json_object"},
        temperature=0.2,
        max_tokens=512,
    )

    content = resp.choices[0].message.content

    # Intentar parsear JSON, si falla usar contenido directo
    try:
        data = json.loads(content)
        borrador = data.get("borrador", "")
    except json.JSONDecodeError:
        # Fallback: tomar todo el content como borrador
        borrador = content.strip()

    # Generar las alertas de plazos
    alertas = _alertas(nuc) if nuc else "—"

    # Formateo final
    return (
        f"## Alertas de plazos\n"
        f"{alertas}\n\n"
        f"## Borrador de fallo\n"
        f"{borrador}"
    )
