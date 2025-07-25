# tools/resumen_doc.py
# ==============================================================
# Genera resúmenes de casos judiciales.
# Lógica:
#   1. Busca sentencia final en los chunks locales.
#   2. Usa colectar_texto(nuc) para otros trámites.
#   3. Si hay final + otros → pregunta "sentencia" o "todo".
#   4. Si solo otros trámites → resume todos con disclaimer.
# ==============================================================
import os, re, json, pandas as pd, difflib
from pathlib import Path
from typing import Dict, List, Tuple
from json import loads, JSONDecodeError
from together import Together

from config import TOGETHER_API_KEY, LLM_MODEL_ID, DATA_DIR
from memory import memory
from trigger_search_documents import colectar_texto

client = Together(api_key=TOGETHER_API_KEY)

# ───────────────────────────── 1. Cargar chunks locales ──────────────────
def _load_chunks() -> Tuple[Dict[str, List[Tuple[int, str]]],
                            Dict[str, List[Tuple[int, str]]]]:
    CHUNKS_DIR = Path(DATA_DIR) / "chunks"
    by_nuc, by_tram = {}, {}
    for p in CHUNKS_DIR.glob("*.json"):
        entry = json.loads(p.read_text(encoding="utf-8"))
        md, text = entry.get("metadata", {}), entry.get("text", "")
        if not text:
            continue
        cid = int(md.get("ChunkID", 0))
        nuc = str(md.get("NUC", "")).lower()
        numt = str(md.get("NumeroTramite", "")).lower()
        if nuc:
            by_nuc.setdefault(nuc, []).append((cid, text))
        if numt:
            by_tram.setdefault(numt, []).append((cid, text))
    for d in (by_nuc, by_tram):
        for k in d:
            d[k].sort(key=lambda x: x[0])
    return by_nuc, by_tram


CHUNKS_BY_NUC, CHUNKS_BY_TRAM = _load_chunks()
print(f"DEBUG resumen_doc: NUCs con sentencia final = {len(CHUNKS_BY_NUC)}")

# ───────────────────────────── 2. Helpers memoria ────────────────────────
# tools/resumen_doc.py  (fragmento relevante)

# ───────────────────────── 1. ESTADO PENDIENTE EN MÓDULO ────────────────
_PENDING = None           #  ←  variable global privada


def _get_pending():
    return _PENDING


def _set_pending(obj: dict):
    global _PENDING
    _PENDING = obj         # se guarda tal cual, sin pasar por LangChain


def _clear_pending():
    global _PENDING
    _PENDING = None
# ─────────────────────────────────────────────────────────────────────────

# ───────────────────────────── 3. Plantilla JSON ─────────────────────────
TEMPLATE = """
Eres un analista experto en sentencias dominicanas.
Lee el texto completo y construye un JSON en una sola línea:

{
  "datos_esenciales": {
    "tribunal": "", "sala": "", "expediente": "", "asunto": "", "fecha": "",
    "numero_tramite": "", "fecha_tramite": ""
  },
  "partes": {
    "demandantes": [ { "nombre": "", "representantes": "" } ],
    "demandados":  [ { "nombre": "", "representantes": "" } ]
  },
  "pretensiones": [""],
  "hechos_probados": "",
  "fundamentos": [""],
  "parte_dispositiva": "",
  "puntos_clave": ""
}

TEXTO SENTENCIA ↓↓↓
"""

def _llm_json(texto: str) -> dict:
    resp = client.chat.completions.create(
        model=LLM_MODEL_ID,
        messages=[{"role": "user", "content": TEMPLATE + texto}],
        response_format={"type": "json_object"},
        temperature=0.0,
        max_tokens=1500,
    )
    return loads(resp.choices[0].message.content)

def _md_from_data(d: dict) -> str:
    def _list(l): return "\n".join(
        f"- **{p['nombre']}** (repr.: {p['representantes']})"
        for p in l if p["nombre"]) or "- —"
    es = d["datos_esenciales"]
    md  = "### 1. Datos esenciales\n" + \
          "\n".join(f"- **{k.capitalize()}**: {v}" for k, v in es.items() if v)
    md += "\n\n### 2. Partes\n**Demandantes:**\n" + _list(d["partes"]["demandantes"])
    md += "\n\n**Demandados:**\n" + _list(d["partes"]["demandados"])
    md += "\n\n### 3. Pretensiones\n" + "\n".join(f"1. {p}" for p in d["pretensiones"])
    md += "\n\n### 4. Hechos probados\n" + d["hechos_probados"]
    md += "\n\n### 5. Fundamentos jurídicos\n" + \
          "\n".join(f"- {f}" for f in d["fundamentos"])
    md += "\n\n### 6. Parte dispositiva\n" + d["parte_dispositiva"]
    md += "\n\n> **Puntos clave:** " + d["puntos_clave"]
    return md

# ───────────────────────────── 4. Función principal ─────────────────────
def run(msg: str) -> str:
    # ── 0. Elección pendiente ───────────────────────────────────────
    pend = _get_pending()
    if pend:
        choice = msg.strip().lower()
        if choice not in ("todo", "sentencia"):
            return "✏️ Responde **sentencia** o **todo**."
        nuc = pend["nuc"]
        df  = pd.DataFrame(pend["df"])
        if choice == "sentencia":
            chunks = CHUNKS_BY_NUC.get(nuc, [])
            if not chunks:
                return "⚠️ No se encontró la sentencia final en los chunks locales."
            data = _llm_json("\n\n".join(t for _, t in chunks))
            _clear_pending()
            return _md_from_data(data)
        # choice == "todo"
        outs = []
        for _, row in df.iterrows():
            txt  = row["texto_pdf"]
            numt = str(row["NumeroTramite"])
            ftra = str(row["FechaCreacion"])
            tag  = f"[TRÁMITE {numt} | {ftra}]"
            d    = _llm_json(f"{tag}\n\n{txt}")
            outs.append(f"## Trámite {numt}\n" + _md_from_data(d))
        _clear_pending()
        return "\n\n---\n\n".join(outs)

    # ── 1. Detectar NUC ─────────────────────────────────────────────
    m = re.search(r"\d{3}-\d{4}-[A-Z0-9]{4}-\d{5}", msg, re.I)
    if not m:
        return "⚠️ Necesito el número de caso (NUC) para acceder a los documentos."
    nuc = m.group(0).lower()

    # ── 2. Sentencia final presente? ────────────────────────────────
    chunks_final = CHUNKS_BY_NUC.get(nuc, [])

    # ── 3. Traer todos los trámites activos ────────────────────────
    try:
        df_docs = colectar_texto(nuc)
        if df_docs is None or df_docs.empty:
            if not chunks_final:
                return f"⚠️ No se encontraron documentos con el NUC “{nuc}”."
            # Solo sentencia final
            data = _llm_json("\n\n".join(t for _, t in chunks_final))
            return _md_from_data(data)
    except Exception as e:
        return f"⚠️ Error al recuperar documentos: {str(e)}"

    # ── 4. Decidir escenario ───────────────────────────────────────
    if chunks_final and len(df_docs) > 1:
        # Escenario A → preguntar
        _set_pending({"nuc": nuc, "df": df_docs.to_dict()})
        return (
            f"🔎 Hay una **sentencia final** y **{len(df_docs)}** trámites activos.\n"
            "Escribe **sentencia** para resumir solo la sentencia final, o "
            "**todo** para resumir todos los documentos."
        )

    if not chunks_final:
        # Escenario B → disclaimer y resumir todo
        outs = []
        for _, row in df_docs.iterrows():
            txt  = row["texto_pdf"]
            numt = str(row["NumeroTramite"])
            ftra = str(row["FechaCreacion"])
            tag  = f"[TRÁMITE {numt} | {ftra}]"
            d    = _llm_json(f"{tag}\n\n{txt}")
            outs.append(f"## Trámite {numt}\n" + _md_from_data(d))
        disclaimer = (
            f"⚠️ No se encontró una sentencia final registrada para el caso **{nuc.upper()}**.\n"
            f"Se genera resumen de **{len(df_docs)}** documentos del proceso activo:\n\n"
        )
        return disclaimer + "\n\n---\n\n".join(outs)

    # Escenario A pero solo sentencia final (único trámite adicional)
    data = _llm_json("\n\n".join(t for _, t in chunks_final))
    return _md_from_data(data)
