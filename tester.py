# legal_taxonomy.py  (versión con fallback robusto)
import re, json, requests, xml.etree.ElementTree as ET
from pathlib import Path
from unidecode import unidecode

OUT      = Path("legal_concepts.json")
LOCAL_WN = Path("wordnet_spa.xml")        # <- copia local opcional
DPEJ_URL = "https://dpej.rae.es/lemas/"
WN_URL   = "https://adimen.si.ehu.es/web/MCR/versions/3.0/wordnet_spa.xml"

def _norm(t: str) -> str:
    return unidecode(t.lower().strip())

def _download(url: str, path: Path, timeout: int = 60) -> bool:
    """Descarga *url* a *path*; devuelve True si tuvo éxito."""
    try:
        r = requests.get(url, timeout=timeout)
        r.raise_for_status()
        path.write_bytes(r.content)
        return True
    except Exception as e:
        print(f"⚠️  No pude descargar {url}: {e}")
        return False

def build_bank(force: bool = False) -> dict[str, list[str]]:
    if OUT.exists() and not force:
        return json.loads(OUT.read_text())

    concepts: dict[str, set[str]] = {}

    # ── 1. Diccionario Panhispánico (scrape rápido) ──────────────────
    for letter in "abcdefghijklmnopqrstuvwxyz":
        try:
            html = requests.get(f"{DPEJ_URL}{letter}", timeout=30).text
            for m in re.finditer(r'class="list-group-item[^"]*">([^<]+)</a>', html):
                lemma = _norm(m.group(1))
                concepts.setdefault(lemma, set()).add(lemma)
        except Exception as e:
            print(f"⚠️  Falló la conexión al DPEJ ({letter}): {e}")
            break  # aborta scraping masivo; evita bucle

    # ── 2. Spanish Legal WordNet (con fallback local) ────────────────
    if not LOCAL_WN.exists():
        _download(WN_URL, LOCAL_WN)

    if LOCAL_WN.exists():
        try:
            root = ET.fromstring(LOCAL_WN.read_bytes())
            for synset in root.findall(".//Synset"):
                words = [_norm(w.get("lemma")) for w in synset.findall("./Word")]
                for w in words:
                    concepts.setdefault(w, set()).update(words)
        except ET.ParseError as e:
            print(f"⚠️  No pude parsear WordNet ({e}); continuando sin él.")
    else:
        print("⚠️  No hay copia local de Spanish Legal WordNet; "
              "el banco tendrá solo los lemas del DPEJ.")

    # ── 3. Serializar ────────────────────────────────────────────────
    OUT.write_text(json.dumps({k: sorted(v) for k, v in concepts.items()},
                              ensure_ascii=False, indent=2))
    print(f"✅ Banco creado con {len(concepts):,} conceptos.")
    return json.loads(OUT.read_text())

# acceso global
BANK = build_bank()
