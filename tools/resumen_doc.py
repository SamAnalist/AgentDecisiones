# tools/resumen_doc.py — versión sin chunks, resume todo lo de trigger_document_search
# ==============================================================
# Genera resúmenes de casos judiciales **sin** depender de chunks locales.
# Flujo:
#   1) Extrae NUC del mensaje o del historial.
#   2) Llama a colectar_texto(nuc) para traer TODOS los documentos.
#   3) Heurística para identificar si hay "sentencia final".
#   4) Si hay final y además otros trámites → pregunta al usuario
#      si quiere **todo** o **solo la sentencia** (con estado PENDING liviano).
#   5) Si no hay ambigüedad → resume directamente (todo o solo la final).
# ==============================================================
from __future__ import annotations
import re, json, pandas as pd
from typing import Optional, List, Tuple, Dict
from json import loads
from together import Together

from config import TOGETHER_API_KEY, LLM_MODEL_ID
from memory import memory
from trigger_search_documents import colectar_texto

client = Together(api_key=TOGETHER_API_KEY)

# ───────────────────────── utilidades de memoria/chat ────────────────────
def _get_history_text() -> str:
    try:
        return (memory.load_memory_variables({}).get("history") or "").strip()
    except Exception:
        return ""

# ───────────────────────── extracción de señales ─────────────────────────
NUC_RE      = re.compile(r"\b\d{3}-\d{4}-[A-Z0-9]{4}-\d{5}\b", re.I)
ALL_RE      = re.compile(r"\b(todos?|todo|completo|entero|todos los documentos|todos los trámites)\b", re.I)
SENT_RE     = re.compile(r"\b(sentencia\s*final|solo\s+la?\s*sentencia|solo\s*sentencia|parte\s+dispositiva|dispositivo|fallo\s*final)\b", re.I)

def _extract_nuc(texts: List[str]) -> Optional[str]:
    for t in texts:
        if not t:
            continue
        m = NUC_RE.search(t)
        if m:
            return m.group(0).lower()
    return None

# ───────────────────────── estado liviano de “pending” ───────────────────
# Guardamos solo lo mínimo para no cargar memoria con textos.
_PENDING: Optional[Dict] = None

def _set_pending(nuc: str, n_docs: int):
    global _PENDING
    _PENDING = {"nuc": nuc, "n_docs": n_docs}

def _pop_pending() -> Optional[Dict]:
    global _PENDING
    p = _PENDING
    _PENDING = None
    return p

# ───────────────────────── detección de sentencia final ──────────────────
FINAL_HARD_HINT_COLS = ("TipoFallo", "TipoDocumento", "Tipo", "Clase")
FINAL_HARD_HINT_VALS = ("sentencia", "sentencia definitiva", "fallo", "dispositivo")

FINAL_RX = re.compile(
    r"\b(POR\s+TALES\s+MOTIVOS|PARTE\s+DISPOSITIVA|DISPOSITIVO|"
    r"Y\s+POR\s+NUESTRA\s+SENTENCIA|SE\s+PRONUNCIA|SE\s+ORDENA)\b",
    re.I
)

def _final_score(row: pd.Series) -> int:
    score = 0
    # pista por columnas (si existen)
    for c in FINAL_HARD_HINT_COLS:
        val = str(row.get(c, "") or "").lower()
        if any(v in val for v in FINAL_HARD_HINT_VALS):
            score += 3
            break
    # pista por texto
    txt = str(row.get("texto_pdf") or row.get("textoPDF") or "")
    if FINAL_RX.search(txt):
        score += 3
    # presencia de “PRIMERO: / SEGUNDO: / TERCERO: …” suma
    score += len(re.findall(r"\b(PRIMERO|SEGUNDO|TERCERO|CUARTO)\s*:", txt, flags=re.I))
    return score

def _pick_final_row(df: pd.DataFrame) -> Optional[pd.Series]:
    if df is None or df.empty:
        return None
    # normaliza posible nombre de columna de texto
    if "texto_pdf" not in df.columns and "textoPDF" in df.columns:
        df = df.rename(columns={"textoPDF": "texto_pdf"})
    # elige la fila con mayor “final_score” y umbral mínimo
    df = df.copy()
    df["_score__final"] = df.apply(_final_score, axis=1)
    df = df.sort_values("_score__final", ascending=False)
    best = df.iloc[0]
    return best if best["_score__final"] >= 3 else None

# ───────────────────────── LLM: JSON + Markdown ──────────────────────────
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

MAX_CHARS_PER_SUMMARY = 120_000  # recorte duro por si el texto es enorme

def _llm_json(texto: str) -> dict:
    if MAX_CHARS_PER_SUMMARY and len(texto) > MAX_CHARS_PER_SUMMARY:
        texto = texto[:MAX_CHARS_PER_SUMMARY]
    resp = client.chat.completions.create(
        model=LLM_MODEL_ID,
        messages=[{"role": "user", "content": TEMPLATE + texto}],
        response_format={"type": "json_object"},
        temperature=0.0,
        max_tokens=1500,
    )
    return loads(resp.choices[0].message.content)

def _md_from_data(d: dict, titulo: Optional[str] = None) -> str:
    def _list(l):
        return "\n".join(
            f"- **{(p or {}).get('nombre','')}** (repr.: {(p or {}).get('representantes','')})"
            for p in (l or []) if (p or {}).get("nombre")
        ) or "- —"
    es = d.get("datos_esenciales", {})
    md  = (f"## {titulo}\n\n" if titulo else "")
    md += "### 1. Datos esenciales\n" + \
          "\n".join(f"- **{k.capitalize()}**: {v}" for k, v in es.items() if v)
    partes = d.get("partes", {"demandantes":[], "demandados":[]})
    md += "\n\n### 2. Partes\n**Demandantes:**\n" + _list(partes.get("demandantes",[]))
    md += "\n\n**Demandados:**\n" + _list(partes.get("demandados",[]))
    pret = d.get("pretensiones", [])
    md += "\n\n### 3. Pretensiones\n" + ("\n".join(f"1. {p}" for p in pret) if pret else "- —")
    md += "\n\n### 4. Hechos probados\n" + (d.get("hechos_probados","") or "—")
    fnds = d.get("fundamentos", [])
    md += "\n\n### 5. Fundamentos jurídicos\n" + ("\n".join(f"- {f}" for f in fnds) if fnds else "- —")
    md += "\n\n### 6. Parte dispositiva\n" + (d.get("parte_dispositiva","") or "—")
    md += "\n\n> **Puntos clave:** " + (d.get("puntos_clave","") or "—")
    return md

# ───────────────────────── función principal ─────────────────────────────
def run(msg: str) -> str:
    # 0) ¿respuesta a un pending?
    pend = _pop_pending()
    if pend:
        choice = msg.strip().lower()
        if choice not in ("todo", "sentencia"):
            # si la respuesta no es válida, re-arma el pending
            _set_pending(pend["nuc"], pend["n_docs"])
            return "✏️ Escribe **sentencia** para resumir solo la sentencia final, o **todo** para resumir todos los documentos."
        nuc = pend["nuc"]
        df  = colectar_texto(nuc)
        if df is None or df.empty:
            return f"⚠️ No se encontraron documentos para el NUC **{nuc.upper()}**."
        # normaliza texto
        if "texto_pdf" not in df.columns and "textoPDF" in df.columns:
            df = df.rename(columns={"textoPDF": "texto_pdf"})
        final_row = _pick_final_row(df)
        if choice == "sentencia":
            if final_row is None:
                # si no hay final, ofrecemos todo
                outs = _summarize_all(df)
                return (f"⚠️ No se detectó una sentencia final para **{nuc.upper()}**. "
                        f"Se generan resúmenes de **{len(outs)}** documentos activos.\n\n"
                        + "\n\n---\n\n".join(outs))
            data = _llm_json(str(final_row.get("texto_pdf","")))
            return _md_from_data(data, titulo="Sentencia final")
        # choice == "todo"
        outs = _summarize_all(df)
        return "\n\n---\n\n".join(outs)

    # 1) NUC desde msg o history
    history = _get_history_text()
    nuc = _extract_nuc([msg, history])
    if not nuc:
        return "⚠️ Necesito el número de caso (NUC). Escríbelo así: 034-2021-ECON-00068."

    # 2) Traer TODOS los documentos (trigger_document_search)
    df_docs = colectar_texto(nuc)
    if df_docs is None or df_docs.empty:
        return f"⚠️ No se encontraron documentos para el NUC **{nuc.upper()}**."

    # normaliza texto
    if "texto_pdf" not in df_docs.columns and "textoPDF" in df_docs.columns:
        df_docs = df_docs.rename(columns={"textoPDF": "texto_pdf"})

    # 3) ¿el usuario ya indicó alcance?
    wants_all  = bool(ALL_RE.search(msg) or ALL_RE.search(history))
    wants_sent = bool(SENT_RE.search(msg) or SENT_RE.search(history))

    # 4) Heurística de sentencia final
    final_row = _pick_final_row(df_docs)
    has_final = final_row is not None
    many_docs = len(df_docs) > 1

    # 5) Resolución de alcance
    if wants_all:
        outs = _summarize_all(df_docs)
        return "\n\n---\n\n".join(outs)

    if wants_sent:
        if has_final:
            data = _llm_json(str(final_row.get("texto_pdf","")))
            return _md_from_data(data, titulo="Sentencia final")
        # pidió sentencia pero no hay final → resume todo
        outs = _summarize_all(df_docs)
        return (f"⚠️ No se detectó una sentencia final para **{nuc.upper()}**. "
                f"Se generan resúmenes de **{len(outs)}** documentos activos.\n\n"
                + "\n\n---\n\n".join(outs))

    # No indicó alcance: si hay final y además más trámites, **pregunta**
    if has_final and many_docs:
        _set_pending(nuc, len(df_docs))
        return (f"🔎 Hay una **sentencia final** y **{len(df_docs)}** trámites/documentos activos en **{nuc.upper()}**.\n"
                "Escribe **sentencia** para resumir solo la sentencia final, o **todo** para resumir todos.")

    # Si no hay ambigüedad: decide y devuelve
    if has_final and not many_docs:
        data = _llm_json(str(final_row.get("texto_pdf","")))
        return _md_from_data(data, titulo="Sentencia final")

    # sin final → resume todos
    outs = _summarize_all(df_docs)
    disclaimer = (f"⚠️ No se detectó una sentencia final para **{nuc.upper()}**. "
                  f"Se generan resúmenes de **{len(outs)}** documentos activos.\n\n")
    return disclaimer + "\n\n---\n\n".join(outs)

# ───────────────────────── helpers internos ──────────────────────────────
def _summarize_all(df_docs: pd.DataFrame) -> List[str]:
    outs: List[str] = []
    for _, row in df_docs.iterrows():
        txt = str(row.get("texto_pdf", "") or "")
        if not txt.strip():
            continue
        numt = str(row.get("NumeroTramite","") or "")
        ftra = str(row.get("FechaTramite","") or "")
        tag  = f"[TRÁMITE {numt or 's/n'} | {ftra or 's/f'}]"
        d    = _llm_json(f"{tag}\n\n{txt}")
        outs.append(_md_from_data(d, titulo=f"Trámite {numt or 's/n'}"))
    return outs
