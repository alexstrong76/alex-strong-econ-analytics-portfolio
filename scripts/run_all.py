# =========================================
# File: scripts/run_all.py            (orchestrator)
# =========================================
"""
Runs all generator scripts under scripts/*.py (excluding itself),
captures logs, enforces timeouts, and gathers CSVs to data/processed/.
"""
from __future__ import annotations
import os, shlex, sys, time, shutil, subprocess
from pathlib import Path
from typing import List

SCRIPTS_DIR = Path("scripts")
OUTPUT_DIR = Path(os.getenv("OUTPUT_DIR", "data/processed"))
LOG_DIR = Path("artifacts/logs")
TIMEOUT_SEC = int(os.getenv("SCRIPT_TIMEOUT_SEC", "180"))  # per script

EXCLUDE = {"run_all.py", "__init__.py"}

def discover_scripts() -> List[Path]:
    return sorted(p for p in SCRIPTS_DIR.glob("*.py") if p.name not in EXCLUDE)

def run_script(path: Path) -> int:
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    log_file = LOG_DIR / f"{path.stem}.log"

    env = os.environ.copy()
    env.setdefault("OUTPUT_DIR", str(OUTPUT_DIR))

    cmds = [
        [sys.executable, str(path), "--outdir", str(OUTPUT_DIR)],  # preferred (if script supports)
        [sys.executable, str(path)],                                # fallback
    ]

    for cmd in cmds:
        with log_file.open("w", encoding="utf-8") as log:
            log.write(f"$ {shlex.join(cmd)}\n\n")
            log.flush()
            try:
                t0 = time.time()
                proc = subprocess.run(
                    cmd, env=env, stdout=log, stderr=log, timeout=TIMEOUT_SEC, check=False
                )
                dt = time.time() - t0
                log.write(f"\n---- exit={proc.returncode} elapsed={dt:.2f}s ----\n")
                log.flush()
                if proc.returncode == 0:
                    return 0
            except subprocess.TimeoutExpired as e:
                log.write(f"\n---- TIMEOUT after {TIMEOUT_SEC}s ----\n{e}\n")
                log.flush()
                return 124  # timeout code
    return 1  # failed both attempts

def gather_csvs() -> int:
    """Ensure all CSVs live under OUTPUT_DIR; copy any stray CSVs."""
    moved = 0
    for csv_path in Path(".").glob("**/*.csv"):
        # skip desired target, VCS, venvs, cache
        rel = csv_path.as_posix()
        if rel.startswith(".git/") or "/.venv/" in rel or "/venv/" in rel or rel.startswith("artifacts/"):
            continue
        if OUTPUT_DIR in csv_path.parents or csv_path.parent == OUTPUT_DIR:
            continue
        # copy into OUTPUT_DIR, preserving basename; avoid overwrite by suffixing
        dest = OUTPUT_DIR / csv_path.name
        if dest.exists():
            stem, suf = dest.stem, dest.suffix
            i = 1
            while dest.exists():
                dest = OUTPUT_DIR / f"{stem}_{i}{suf}"
                i += 1
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(csv_path, dest)
        moved += 1
    print(f"CSV gather complete. Moved/copied: {moved}")
    return moved

def main() -> int:
    scripts = discover_scripts()
    if not scripts:
        print("No scripts found in scripts/*.py")
        return 0
    print(f"Discovered {len(scripts)} script(s): " + ", ".join(p.name for p in scripts))
    failures = []
    for p in scripts:
        code = run_script(p)
        status = "OK" if code == 0 else f"FAIL({code})"
        print(f"{p.name}: {status}")
        if code != 0:
            failures.append((p.name, code))
    gather_csvs()
    if failures:
        print("Some scripts failed:")
        for name, code in failures:
            print(f" - {name}: exit {code}")
        # return non-zero to make CI red; change to 0 to keep green
        return 1
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
