# =========================================
# File: scripts/generate_csv.py       (example 1)
# =========================================
"""
Writes demo CSV to OUTPUT_DIR (default: data/processed).
WHY: Standardizes outputs for CI collection.
"""
from __future__ import annotations
import csv, os
from pathlib import Path

def main() -> int:
    out_dir = Path(os.getenv("OUTPUT_DIR", "data/processed"))
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / "demo_metrics.csv"
    rows = [
        {"id": 1, "name": "alpha", "value": 3.14},
        {"id": 2, "name": "beta",  "value": 2.71},
        {"id": 3, "name": "gamma", "value": 1.62},
    ]
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=rows[0].keys())
        w.writeheader(); w.writerows(rows)
    print(f"Wrote {path} ({len(rows)} rows)")
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
