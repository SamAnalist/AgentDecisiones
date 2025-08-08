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
    "relacionar_juris",
    "comparar_ids",          # ← NUEVO
    "consulta_concepto",
    "borrador_alerta",
    "cronologia",
    "auditoria_ley",
    "conversacional",
)


client = Together(api_key=TOGETHER_API_KEY)
EXAMPLES = [

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
{"role": "assistant", "content": "estadistica"},
{"role": "user", "content": "audita la sentencia y dime qué artículos omitieron"},
{"role": "assistant", "content": "auditoria_ley"},
{"role": "user", "content": "[PDF:pdf@1234abcd] Resume el documento"},
{"role": "assistant", "content": "resumen_doc"},
{"role": "user", "content": "Compara 533-2020-ECON-00601 con 034-2019-ECON-01057"},
    {"role": "assistant", "content": "comparar_ids"},
{"role": "user", "content": "comparame los documentos cuyo NUCs son: 533-2020-Adm-00241 y 037-2020-ECIV-00195"},
    {"role": "assistant", "content": "comparar_ids"},
# {"role": "user", "content": "Relaciona dos casos similares al PDF que te mandé"},
# {"role": "assistant", "content": "relacionar_juris"},
{"role":"user", "content":"Compárame los dos documentos"},
{"role":"assistant","content":"comparar_ids"},

# --- 10 pares extra enfocados en temas jurídicos generales (↦ conversacional) ----
{"role": "user", "content": "¿Qué pasos debe seguir un ciudadano para presentar un recurso de amparo en la República Dominicana?"},
{"role": "assistant", "content": "conversacional"},

{"role": "user", "content": "Resume en pocas líneas la jurisprudencia internacional sobre el derecho al olvido digital."},
{"role": "assistant", "content": "conversacional"},

{"role": "user", "content": "¿Cómo se eligen los jueces de la Corte Interamericana de Derechos Humanos?"},
{"role": "assistant", "content": "conversacional"},

{"role": "user", "content": "Explica el impacto de la Ley 63-17 de Tránsito en la reducción de accidentes viales."},
{"role": "assistant", "content": "conversacional"},

{"role": "user", "content": "Compara brevemente el sistema inquisitivo y el acusatorio en Latinoamérica."},
{"role": "assistant", "content": "conversacional"},

{"role": "user", "content": "suministre los 5 divorcios solo por incompatibilidad de caracteres"},
{"role": "assistant", "content": "conversacional"},

{"role": "user", "content": "Dame un ejemplo de precedente vinculante emitido por el Tribunal Constitucional español y su relevancia."},
{"role": "assistant", "content": "conversacional"},

{"role": "user", "content": "¿Cuál es el límite máximo de prisión preventiva según los estándares de la Corte IDH?"},
{"role": "assistant", "content": "conversacional"},

{"role": "user", "content": "¿Qué opinas sobre la reciente reforma del Código Penal peruano respecto al feminicidio?"},
{"role": "assistant", "content": "conversacional"},

{"role": "user", "content": "Explícame cómo la doctrina de la 'excepción de inconstitucionalidad' se aplica en Colombia."},
{"role": "assistant", "content": "conversacional"},
{"role": "user", "content": "Encuentra N sentencias sobre divorcio"},
{"role": "assistant", "content": "conversacional"},
{"role": "user", "content": "Encuentrame N sentencias sobre robo"},
{"role": "assistant", "content": "conversacional"},

]
SYSTEM_PROMPT = (
    "Eres un asistente jurídico que CLASIFICA la intención de una pregunta. Además, la pregunta debe ser de indole judicial/legal o refiriendose a documentos y leyes.\n"
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
    "- estadistica_ai → pide conteo/estadística que requieren busqueda semantica del concepto dentro del texto."
    "  'porcentaje de demandas contra EDESUR'.  Estas consultas requieren "
    "  búsqueda semántica en los **embeddings** de textoPDF.\n"
    "- relacionar_juris → Pide un veredicto recomendado o se refiere a describe un caso pidiendo recomendación de fallo.\n"
    "- borrador_alerta → plazos, vencimientos, alertas procesales\n"
    "- auditoria_ley      –  solicita verificar artículos citados/omitidos o "
    "                        auditar un borrador de fallo frente al corpus legal\n"
    "- consulta_concepto → definiciones y aclaraciones jurídicas puntuales.\n"
    "- conversacional    **Cualquier pregunta que no aplique para otro query** (preguntas abiertas, pedidos de diferentes casos, etc.)\n\n"
     "- comparar_ids   → compara dos sentencias de la BD dadas por sus IDs\n"
    "Devuelve solo la etiqueta, sin explicaciones."
)

def detect_intent(msg: str) -> Literal[
    "expediente",
    "resumen_doc",
    "consulta_doc",
    "estadistica",
    "relacionar_juris",
    "consulta_concepto",
    "auditoria_ley",
    "comparar_ids",
    "borrador_alerta",
    "cronologia",
    "conversacional"
]:
    response = client.chat.completions.create(
        model=LLM_MODEL_ID,
        messages=[{"role": "system", "content": SYSTEM_PROMPT}, *EXAMPLES,
                  {"role": "user", "content": msg}],
        temperature=0.0,
    )
    label = response.choices[0].message.content.strip().lower()
    #print("Ve aqui lo que recibe y envia el router: ",msg,"", label)
    if label in LABELS:
        return label
    return "conversacional"
