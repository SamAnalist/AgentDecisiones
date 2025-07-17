import pathlib, sys, numpy as np
ROOT = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

import vectorstore
from embed import BNEEmbeddings

embedder = BNEEmbeddings()
idx      = vectorstore.vectordb.index       # faiss.IndexFlatL2
terms    = ["robo", "manutención", "divorcio"]

for term in terms:
    vec = np.asarray(embedder.embed_query(term), dtype="float32").reshape(1, -1)
    dists, _ = idx.search(vec, 10)
    print(f"\n▷ {term}: {[round(d,3) for d in dists[0]]}")
