from together import Together
from config import TOGETHER_API_KEY, LLM_MODEL_ID, K_RETRIEVE, SIM_THRESHOLD
from vectorstore import search_by_vector
from embed import BNEEmbeddings

client = Together(api_key=TOGETHER_API_KEY)

def run(msg: str) -> str:
    emb = BNEEmbeddings()
    q_vec = emb.embed_query(msg)
    docs = search_by_vector(q_vec, k=K_RETRIEVE * 4)

    if not docs:
        return "⚠️ No hallé precedentes relevantes."

    # Filtrar por score alto primero
    buenos = [d for d in docs if d.metadata.get("score", 1) >= SIM_THRESHOLD]
    if len(buenos) < K_RETRIEVE:
        buenos += [d for d in docs if d not in buenos][:K_RETRIEVE - len(buenos)]

    context = "\n".join([d.page_content for d in buenos[:K_RETRIEVE]])

    prompt = (
        "Eres un asistente de juez de República Dominicana, recibirás un número determinado de casos que pueden parecerse al input del usuario, debes hacer una comparación entre la entrada y el contexto."
        "Elige los 2 casos que más se parecen al caso que propone el usuario."
        "Devuelve un texto con la forma:\n"
        """  Caso_id_1: Un resumen del caso \n"""
        """  Caso_id_2: Un resumen del caso \n"""
        """  Veredicto: Escribe aquí en qué se parecen los casos y una recomendación simulando un veredicto de un juez."""
        "Devuelve un texto solo con lo pedido."
        "Estos son los casos encontrados más parecidos al presentado.\n"
        + context
    )

    try:
        resp = client.chat.completions.create(
            model=LLM_MODEL_ID,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.4,
            max_tokens=1024,
        )
        data = resp.choices[0].message.content
        if data:
        # Normalizar precedentes que pueden venir como lista o dict
            return data
        else:
            return "⚠️ Formato inesperado de los precedentes."

    except Exception as e:
        return f"⚠️ Error al generar comparación: {str(e)}"
