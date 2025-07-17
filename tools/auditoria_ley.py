import os
import pickle
from pathlib import Path
import numpy as np
import faiss
from together import Together
from config import (
    TOGETHER_API_KEY,
    TOGETHER_BASE_URL,
    INDEX_DIR,
    LLM_MODEL_ID,
    K_RETRIEVE,
    SIM_THRESHOLD
)
from embed import get_embeddings

# === Configuración de Together AI ===
if not TOGETHER_API_KEY:
    raise EnvironmentError("TOGETHER_API_KEY no está definida en el entorno")
client = Together(api_key=TOGETHER_API_KEY, base_url=TOGETHER_BASE_URL)

# === Carga de índice de leyes ===
idx_dir = Path(INDEX_DIR) / "index_laws"
index_path = idx_dir / "index.faiss"
meta_path = idx_dir / "index.pkl"
if not index_path.exists() or not meta_path.exists():
    raise FileNotFoundError(f"Índice o metadata no encontrado en {idx_dir}")
index = faiss.read_index(str(index_path))
with open(meta_path, "rb") as f:
    metadata = pickle.load(f)  # Lista de fragmentos de texto de leyes

# Embedding para leyes
embedder = get_embeddings(kind="laws")

# === Función de búsqueda semántica ===
def msearch(query: str, k: int = K_RETRIEVE):
    """
    Devuelve hasta k fragmentos de ley más relevantes a la query,
    filtrando por SIM_THRESHOLD.
    """
    # Generar embedding de la consulta
    if hasattr(embedder, "embed_query"):
        q_vec = embedder.embed_query(query)
    else:
        q_vec = embedder.encode([query])[0]
    q_arr = np.array([q_vec], dtype="float32")

    # Búsqueda en FAISS
    distances, indices = index.search(q_arr, k)
    results = []
    for dist, idx in zip(distances[0], indices[0]):
        score = float(dist)
        if score < SIM_THRESHOLD:
            continue
        results.append({"text": metadata[idx], "score": score})
    return results

# === Función de consulta con Llama-4 ===
def ask_llama4(query: str, k: int = K_RETRIEVE, max_tokens: int = 300) -> str:
    """
    Recupera contexto con msearch y genera respuesta usando Llama-4.
    """
    fragments = msearch(query, k=k)
    prompt_lines = [
        "Eres un asistente legal experto en auditoría de leyes. A continuación fragmentos relevantes:\n"
    ]
    for frag in fragments:
        prompt_lines.append(f'"""\n{frag["text"]}\n"""\n')
    prompt_lines.append(f"Pregunta: {query}\nRespuesta:")
    prompt = "\n".join(prompt_lines)

    response = client.completions.create(
        model=LLM_MODEL_ID,
        prompt=prompt,
        max_tokens=max_tokens,
        temperature=0.0
    )
    return response["choices"][0]["text"].strip()

# === Punto de entrada run ===
def run(query: str, top_k: int = None, max_tokens: int = None) -> str:
    """
    Ejecuta pipeline RAG para auditoría de leyes:
    - msearch sobre índice de leyes
    - llama a Llama-4
    """
    k = top_k if top_k is not None else K_RETRIEVE
    mt = max_tokens if max_tokens is not None else 300
    return ask_llama4(query=query, k=k, max_tokens=mt)
