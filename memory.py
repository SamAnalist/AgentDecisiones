"""
memory.py
---------

Buffer de conversación global. Almacena los últimos 12 turnos
(usuario ↔ asistente) para mantener el contexto durante la charla
sobre un mismo caso.  Si quieres más “memoria” basta con aumentar k.

En el futuro, si se requiere memoria a largo plazo o almacenamiento
en base de datos, se puede sustituir ConversationBufferMemory por
un Memory personalizado (por ejemplo, un vectorstore dedicado
o una tabla SQL con claves de sesión).
"""

from langchain.memory import ConversationBufferMemory

memory = ConversationBufferMemory(
    k=12,
    return_messages=True,
)

