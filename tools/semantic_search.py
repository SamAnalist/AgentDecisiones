"""
tools/semantic_search.py
 Buscador directo sobre FAISS que devuelve (Document, distancia L2)
"""
import sys, pathlib, numpy as np, faiss

ROOT = pathlib.Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import vectorstore

_idx        = vectorstore.vectordb.index                 # faiss.IndexFlatL2
_id2store   = vectorstore.vectordb.index_to_docstore_id  # list[id]
_docstore   = vectorstore.vectordb.docstore              # InMemoryDocstore

def search_with_scores(vec: list[float], k: int = 10, filtro=None):
    q = np.asarray(vec, dtype="float32").reshape(1, -1)
    dists, idxs = _idx.search(q, k)
    results = []
    for dist, ix in zip(dists[0], idxs[0]):
        if ix == -1:
            continue
        doc_id = _id2store[ix]
        doc    = _docstore._dict[doc_id]     # acceso interno
        if filtro:
            meta = doc.metadata or {}
            if not all(meta.get(k) == v for k, v in filtro.items()):
                continue
        results.append((doc, float(dist)))   # menor distancia = mayor similitud
    return results
