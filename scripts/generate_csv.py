# =========================================
# File: scripts/generate_csv.py
# =========================================
"""
Generates a simple CSV into outputs/data.csv using only the Python stdlib.
WHY: Avoids dependency issues; runs anywhere without extra installs.
"""
from __future__ import annotations
import csv
from pathlib import Path


def main() -> int:
    out_dir = Path("outputs")
    out_dir.mkdir(parents=True, exist_ok=True)
    out_csv = out_dir / "data.csv"

    rows = [
        {"id": 1, "name": "alpha", "value": 3.14},
        {"id": 2, "name": "beta", "value": 2.71},
        {"id": 3, "name": "gamma", "value": 1.62},
    ]

    with out_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["id", "name", "value"])
        writer.writeheader()
        writer.writerows(rows)

    print(f"Wrote {len(rows)} rows -> {out_csv.resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
