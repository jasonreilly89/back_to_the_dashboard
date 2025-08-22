import json
import os
import random
from datetime import datetime
from pathlib import Path

DEFAULT_METRICS_PATH = "/home/jason/ml/datasets/models/prepped_setup12_60_setup/metrics.jsonl"


def main():
    path = Path(os.environ.get("METRICS_PATH", DEFAULT_METRICS_PATH))
    path.parent.mkdir(parents=True, exist_ok=True)
    day = datetime.now().strftime("%Y%m%d")
    records = []
    for epoch in range(1, 4):
        rec = {
            "day": day,
            "epoch": epoch,
            "train_loss": round(random.uniform(0.5, 1.0), 4),
            "time_s": round(random.uniform(0.5, 3.0), 2),
            "ap_micro": round(random.uniform(0, 1), 4),
            "ap_macro": round(random.uniform(0, 1), 4),
            "prev_micro": round(random.uniform(0, 0.2), 4),
            "prev_macro": round(random.uniform(0, 0.2), 4),
            "pos_total": random.randint(0, 2000),
            "neg_total": random.randint(1000, 5000),
        }
        records.append(rec)
    with path.open("a", encoding="utf-8") as f:
        for rec in records:
            f.write(json.dumps(rec) + "\n")
    print(f"Wrote {len(records)} rows to {path}")


if __name__ == "__main__":
    main()
