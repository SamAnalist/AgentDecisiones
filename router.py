# router.py — solo LLM, sin reglas manuales
import streamlit
from together import Together
from typing import Literal
from config import TOGETHER_API_KEY, LLM_MODEL_ID
LABELS = (
    "expediente",
    "resumen_doc",
    "consulta_doc",   # ← NUEVO label
    "estadistica",
    "estadistica_ai",
    "comparar_juris",
    "consulta_concepto",
    "borrador_alerta",
    "cronologia",
    #"auditoria_ley",
    "desconocido",
)


client = Together(api_key=TOGETHER_API_KEY)
EXAMPLES = [
{"role": "user", "content": "¿Cuántos casos de robo hay en la base?"},
{"role": "assistant", "content": "estadistica_ai"},
{"role": "user", "content": "Promedio de indemnizaciones por homicidio 2023"},
{"role": "assistant", "content": "estadistica_ai"},
{"role": "user", "content": "¿Cuántas sentencias dictó la Segunda Sala en 2024?"},
{"role": "assistant", "content": "estadistica"},
{"role": "user", "content": "Resume el caso de numero de caso 532-2020-ECON-00560"},
{"role": "assistant", "content": "resumen_doc"},
{"role": "user", "content": "Escribe un expediente del caso de numero de caso 532-2020-ECON-00560"},
{"role": "assistant", "content": "expediente"},
{"role": "user", "content": "sentencia"},
{"role": "assistant", "content": "resumen_doc"},
{"role": "user", "content": "todo"},
{"role": "assistant", "content": "resumen_doc"},
{"role": "user", "content": "Dame la cronología completa del caso 034-2021-ECON-00366"},
{"role": "assistant", "content": "cronologia"},
{"role": "user", "content": "Cuantos casos cuyo fallo acoge en 2021?"},
{"role": "assistant", "content": "estadistica"}
]
SYSTEM_PROMPT = (
    "Eres un asistente jurídico que CLASIFICA la intención de una pregunta.\n"
    "Las preguntas pueden referirse a:\n"
    "• El **DataFrame** principal de contexto y de donde el usuario hará preguntas que tiene columnas: "
    "  ['NUC', 'NumeroTramite', 'IdDocumento', 'Sala', 'Tribunal', "
    "   'Materia', 'TipoFallo', 'FechaDecision', 'textoPDF', ...]\n"
    "• Sentencias activas (texto completo en memoria)\n\n"
    "Devuelve SOLO UNA de estas etiquetas:\n"
    "- expediente → El usuario consulta un expediente sobre un caso con un identificador\n"
    "- resumen_doc → El usuario pide explicitamente un resumen de un documento citado por id o si mencionan algo parecido a: *todo* o *solo la sentencia final*\n"
    "- consulta_doc → pregunta detalles del documento activo\n"
    "- cronologia → solicita la línea de tiempo completa de un caso (NUC)\n"      # ← NUEVO
    "- estadistica → *la métrica se puede calcular SOLO con columnas "
    "  explícitas del DataFrame* (ej.: '¿cuántos fallos en 2023?', "
    "  'promedio de indemnización en Materia = Laboral')\n"
    "- estadistica_ai → pide conteo/estadística sobre conceptos que NO son "
    "  columnas directas (delitos, hechos, doctrinas). Ej.: "
    "  '¿cuántos casos de robo?', 'promedio de indemnización por incendio', "
    "  'porcentaje de demandas contra EDESUR'.  Estas consultas requieren "
    "  búsqueda semántica en los **embeddings** de textoPDF.\n"
    "- comparar_juris → busca precedentes o casos similares\n"
    "- borrador_alerta → plazos, vencimientos, alertas procesales\n"
    #"• auditoria_ley      –  solicita verificar artículos citados/omitidos o "
    #"                        auditar un borrador de fallo frente al corpus legal\n"
    "- consulta_concepto → definiciones y aclaraciones jurídicas puntuales.\n"
    "- desconocido → todo lo demás\n"
    "Devuelve solo la etiqueta, sin explicaciones."
)

def detect_intent(msg: str) -> Literal[
    "expediente",
    "resumen_doc",
    "consulta_doc",
    "estadistica",
    "comparar_juris",
    "consulta_concepto",
    #"auditoria_ley",
    "borrador_alerta",
    "cronologia",
    "desconocido"
]:
    response = client.chat.completions.create(
        model=LLM_MODEL_ID,
        messages=[{"role": "system", "content": SYSTEM_PROMPT}, *EXAMPLES,
                  {"role": "user", "content": msg}],
        temperature=0.0,
    )
    label = response.choices[0].message.content.strip().lower()
    print("Ve aqui lo que recibe y envia el router: ",msg,"", label)
    if label in LABELS:
        return label
    return "desconocido"
