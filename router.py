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
    #"borrador_alerta",
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
    # ------------------------------------------------------------------
    #  R O U T I N G   P R O M P T   –   Llama-4-Maverick
    # ------------------------------------------------------------------
"""    Instrucciones generales
    1) Lee la consulta del usuario.
    2) Elige **exactamente una** etiqueta de la lista ⬇️ que mejor describa
       la intención principal.  No devuelvas nada más que la etiqueta.
    3) Si dudás entre dos, elige la más específica.
    4) Usa siempre minúsculas, sin espacios adicionales, sin tildes
       ni puntuación extra.

    Contexto del dominio
    • Todas las consultas se refieren al ámbito jurídico dominicano
      (sentencias, leyes, jurisprudencia, plazos procesales, etc.).
    • El DataFrame principal contiene columnas:
      ['NUC','NumeroTramite','IdDocumento','Sala','Tribunal','Materia',
       'TipoFallo','FechaDecision','textoPDF', …]
    • Puede haber uno o varios *documentos activos* en memoria
      (texto completo de sentencias o escritos).

    Etiquetas posibles
    ──────────────────────────────────────────────────────────────
      conversacional   → Preguntas de charla general o que no encajan
                         en ninguna otra categoría.
      expediente       → Consulta por un expediente o caso específico
                         identificado (NUC, número de trámite, etc.).
      resumen_doc      → Petición explícita de un resumen de un documento
                         concreto (por ID o «solo la sentencia final», «todo»).
      consulta_doc     → Preguntas de detalle sobre el documento activo
                         (partes, argumentos, firmas, etc.).
      cronologia       → Solicita la línea de tiempo completa de un caso.
      estadistica      → Métricas calculables solo con las columnas DEL
                         DataFrame (ej. «¿cuántos fallos en 2023?»).
      estadistica_ai   → Estadísticas que requieren búsqueda semántica
                         en textoPDF (ej. «% de demandas contra EDESUR»).
      relacionar_juris → Explicación de un caso/expediente para obtener
                         recomendación de veredicto o jurisprudencia afín.
      auditoria_ley    → Verificar artículos citados/omitidos o auditar un
                         borrador de sentencia frente al corpus legal.
      consulta_concepto→ Definiciones o aclaraciones jurídicas puntuales.
      comparar_ids     → Comparar dos o más sentencias de la BD por sus IDs.
    IMPORTANTE: Devuelve ÚNICAMENTE la etiqueta seleccionada. Prioriza la etiqueta conversacional."""

)
import math
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
    resp = client.chat.completions.create(
        model=LLM_MODEL_ID,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": msg},
        ],
        temperature=0.0,
        # Devuelve logprobs para la primera palabra — la etiqueta
        logprobs=True, top_logprobs=5, max_tokens=10,
    )

    label = resp.choices[0].message.content.strip().lower()
    logprob = resp.choices[0].logprobs.token_logprobs[0]
    print(resp, logprob)
    # ≈ log p(label)
    confidence = math.exp(logprob)  # 0-1

    # umbral: si la probabilidad es < 35 % → conversacional
    if confidence < 0.1 or label not in LABELS:
        return "conversacional"
    return label