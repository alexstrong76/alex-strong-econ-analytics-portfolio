# =========================================
# File: scripts/generate_panel.py     (example 2)
# =========================================
"""
Second example to show multi-script orchestration.
"""
from __future__ import annotations
import csv, os, random
from pathlib import Path

def main() -> int:
    out_dir = Path(os.getenv("OUTPUT_DIR", "data/processed"))
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / "panel_sim.csv"
    rng = random.Random(42)
    rows = [{"unit": i, "t": t, "y": rng.random()} for i in range(1, 6) for t in range(1, 4)]
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=rows[0].keys())
        w.writeheader(); w.writerows(rows)
    print(f"Wrote {path} ({len(rows)} rows)")
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
