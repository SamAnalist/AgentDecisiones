"""
export_feedback.py
------------------
Convierte `Interactions.xlsx` en dos archivos listos para RLHF:

1.  rewards.jsonl   → formato {prompt, response, score}  (−1, 0, 1)
2.  dpo.jsonl       → formato {prompt, chosen, rejected}  (para DPO)

Ejemplo de uso:
    (.venv) $ python export_feedback.py

Luego se suben a Together / HuggingFace Trainer para fine-tune continuo.
"""

import json
import pandas as pd
from config import DATA_DIR, INTER_FILE, SCORES

OUT_REWARD = DATA_DIR / "rewards.jsonl"
OUT_DPO    = DATA_DIR / "dpo.jsonl"

df = pd.read_excel(INTER_FILE)
if df.empty:
    raise SystemExit("No hay interacciones registradas todavía.")

# 1) rewards.jsonl  --------------------------------------------------
with open(OUT_REWARD, "w", encoding="utf-8") as f:
    for _, row in df.iterrows():
        f.write(json.dumps({
            "prompt":   row["user_msg"],
            "response": row["assistant_msg"],
            "score":    SCORES.get(row["feedback"], 0),
        }, ensure_ascii=False) + "\n")

# 2) dpo.jsonl  ------------------------------------------------------
#   Para cada prompt con al menos un 'Acepta' y un 'Rechaza'
#   se genera un par (chosen, rejected)
grouped = df.groupby("user_msg")
with open(OUT_DPO, "w", encoding="utf-8") as f:
    for prompt, grp in grouped:
        good = grp[grp["feedback"] == "Acepta"]
        bad  = grp[grp["feedback"] == "Rechaza"]
        if not good.empty and not bad.empty:
            f.write(json.dumps({
                "prompt":  prompt,
                "chosen":  good.iloc[0]["assistant_msg"],
                "rejected": bad.iloc[0]["assistant_msg"],
            }, ensure_ascii=False) + "\n")

print("✅ Exportado:", OUT_REWARD, "y", OUT_DPO)
