"""
train_rlhf.py
-------------
Ejemplo mínimo de cómo llamar a Together AI
para entrenar un reward-model con 'rewards.jsonl'.

Requiere: pip install together

Nota: Completa <PROJECT_ID> y <NEW_MODEL_NAME> según tu cuenta.
"""

import together
from config import TOGETHER_API_KEY, DATA_DIR

client = together.Together(api_key=TOGETHER_API_KEY)

job = client.fine_tune.create(
    training_file=str(DATA_DIR / "rewards.jsonl"),
    project_id="<PROJECT_ID>",
    model_id="meta-llama/Llama-4-Maverick-17B-128E-Instruct-FP8",
    n_epochs=1,
    suffix="<NEW_MODEL_NAME>-rm",
    task_type="REWARD_MODEL",
)

print("📡 Lanzado job:", job["id"])
print("Monitorea en dashboard Together AI.")
