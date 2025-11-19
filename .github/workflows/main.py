# File: main.py  (minimal script that writes a CSV)
from __future__ import annotations
import csv
from pathlib import Path

def run() -> int:
    out = Path("outputs"); out.mkdir(parents=True, exist_ok=True)
    path = out / "data.csv"
    rows = [{"id": 1, "name": "alpha"}, {"id": 2, "name": "beta"}]
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["id", "name"])
        w.writeheader(); w.writerows(rows)
    print(f"Wrote {path} ({len(rows)} rows)")
    return 0

if __name__ == "__main__":
    raise SystemExit(run())
# File: requirements.txt (optional if you use only stdlib)
numpy>=1.24,<3.0
