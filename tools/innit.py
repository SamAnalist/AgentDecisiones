"""
tools package
=============

Agrupa las funciones especializadas (queries) que el router puede
invocar.  Cada submódulo expone una función `run(msg: str) -> str`.

Submódulos:
    • expediente.py       → información completa de un expediente/NUC
    • resumen_doc.py      → resumen y considerandos de un IdDocumento
    • estadistica.py      → consultas sobre DataFrame (cuántos casos…)
    • comparar.py         → comparación de jurisprudencia / precedentes
    • borrador_alerta.py  → borrador de fallo + alertas procesales
"""
