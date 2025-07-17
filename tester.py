from langchain_community.vectorstores import FAISS
from langchain.schema import Document
from embed import get_embeddings
from config import INDEX_DIR

# 1) Carga el índice de leyes
lawdb = FAISS.load_local(
    str(INDEX_DIR / "index_laws"),
    embeddings=get_embeddings("laws"),
    allow_dangerous_deserialization=True
)

# 2) Mira cómo se llama la store interna
print("Docstore attrs:", dir(lawdb.docstore))

# 3) Accede al dict interno (habitualmente '_dict')
docs_map = getattr(lawdb.docstore, "_dict", None)
if docs_map is None:
    # en versiones antiguas quizá sea '_docs' o 'docs_map'
    docs_map = getattr(lawdb.docstore, "_docs", None)
if docs_map is None:
    raise RuntimeError("No pude encontrar la colección interna de documentos en lawdb.docstore")

# 4) Número total de sub-chunks
print("Docs en índice:", len(docs_map))

# 5) Ejemplo de metadata y contenido
sample = next(iter(docs_map.values()))
print("Ejemplo metadata:", sample.metadata)
print("Primeros 200 chars:", sample.page_content[:200])
