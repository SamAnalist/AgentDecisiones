# tools/resumen_doc.py

import os
import re
import json
import difflib
from together import Together
from json import loads, JSONDecodeError
from config import TOGETHER_API_KEY, LLM_MODEL_ID, DATA_DIR
from typing import Dict, List
def _to_str(val):
    if isinstance(val, list):
        return " ".join(str(x) for x in val).strip()
    return str(val).strip()

client = Together(api_key=TOGETHER_API_KEY)

# ── Construir índice local de chunks por identificación ───────────
CHUNKS_DIR = os.path.join(DATA_DIR, "chunks")
docs_map: Dict[str, List[str]] = {}

for fname in os.listdir(CHUNKS_DIR):
    if not fname.endswith(".json"):
        continue
    path = os.path.join(CHUNKS_DIR, fname)
    with open(path, "r", encoding="utf-8") as f:
        entry = json.load(f)

    md = entry.get("metadata", {})
    text = entry.get("text", "")

    # Claves candidatas (normalizadas a minúsculas y sin prefijos como “auto:”)
    keys = [
        str(md.get("DocumentID", "") or ""),
        str(md.get("NUC", "") or ""),
        str(md.get("NumeroTramite", "") or ""),
    ]

    for k in keys:
        k_norm = k.lower().lstrip("auto:").strip()
        if k_norm:
            docs_map.setdefault(k_norm, []).append(text)

print(f"DEBUG: chunks cargados = {len(docs_map)} claves")  # ← trazador global
# ────────────────────────────────────────────────────────────────────


def run(msg: str) -> str:
    """
    Dado un mensaje que incluya un DocumentID, NUC o NumeroTramite
    (aunque aparezca dentro de texto o con prefijos), extrae la clave,
    busca en docs_map y devuelve un resumen estructurado.
    Si no hay coincidencia exacta, ofrece hasta 5 sugerencias difusas.
    """

    # 1) Extraer NUC (000-0000-AAAA-00000)
    nuc = re.search(r"\b\d{3}-\d{4}-[A-Z]{4}-\d{5}\b", msg, re.I)
    if nuc:
        key = nuc.group(0)
    else:
        # 2) Extraer NumeroTramite con o sin prefijo
        numt = re.search(r"[A-Za-z]*:?[\dA-Za-z]{1,5}-\d{4}(?:-[A-Za-z0-9]+){1,3}", msg)
        if numt:
            key = numt.group(0)
        else:
            # 3) Tomar número más largo como posible IdDocumento
            nums = re.findall(r"\d+", msg)
            if not nums:
                return "⚠️ No pude identificar un NUC, NumeroTramite o DocumentID en tu consulta."
            key = max(nums, key=len)

    key_norm = key.lower().lstrip("auto:").strip()
    print(f"DEBUG: clave extraída → '{key_norm}'")           # ← trazador

    # — Búsqueda exacta —
    chunks = docs_map.get(key_norm)
    if chunks:
        # resumen_doc.py  ─ dentro de run()

        from tools.consulta_doc import _set_active  # ← importa el setter
        # reemplaza la línea _set_active...
        _set_active({
            "NUC": key_norm,
            "NumeroTramite": key_norm,
            "DocumentID": key_norm
        })
        print("DEBUG: match exacto OK")                       # ← trazador

    # — Búsqueda startswith —
    if not chunks:
        for k_map, chs in docs_map.items():
            if k_map.startswith(key_norm):
                print(f"DEBUG: startswith → '{k_map}'")       # ← trazador
                chunks = chs
                break

    # — Sugerencias difusas si todo falla —
    if not chunks:
        print("DEBUG: sin match; generando sugerencias")      # ← trazador
        suggestions = difflib.get_close_matches(
            key_norm, list(docs_map.keys()), n=5, cutoff=0.3
        )
        if suggestions:
            sug_list = "\n".join(f"- {s}" for s in suggestions)
            return (
                f"⚠️ No encontré una coincidencia exacta para “{key}”.\n"
                "Quizás quisiste decir uno de estos identificadores:\n"
                f"{sug_list}"
            )
        return "⚠️ No encontré ese expediente ni identificadores similares en la Juriteca."

    # Concatenar todos los chunks chunks
    content = "\n\n".join(chunks)

    # Prompt al LLM
    prompt = (
        "Eres un analista profesional de documentos judiciales. "
        "Genera un resumen siguiendo este formato:\n"
        "{resumen: \"\", considerandos: \"\", decision: \"\" }\n\n"
        "Texto fuente:\n\n" + content
    )

    try:
        resp = client.chat.completions.create(
            model=LLM_MODEL_ID,
            messages=[{"role": "user", "content": prompt}],
            response_format={"type": "json_object"},
            temperature=0.0,
            max_tokens=1500,
        )
        data = loads(resp.choices[0].message.content)
        return (
                "**Resumen general del caso:**\n" + _to_str(data.get("resumen", "")) + "\n\n"
                                                                                       "**Considerandos:**\n" + _to_str(
            data.get("considerandos", "")) + "\n\n"
                                             "**Decisión final del juez:**\n" + _to_str(data.get("decision", ""))
        )


    except JSONDecodeError:
        return "⚠️ Hubo un error al interpretar la respuesta del modelo."
    except Exception as e:
        return f"⚠️ Error al generar resumen: {str(e)}"
