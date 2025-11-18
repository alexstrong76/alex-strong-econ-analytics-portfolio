
from pathlib import Path
import nbformat as nbf

MAP = {
    "notebooks/iv_2sls_demo.ipynb": {
        "md": "# IV / 2SLS — Notebook Demo\nWraps the IV example and prints estimates.",
        "src": "econometrics/iv_2sls/iv_2sls.py",
    },
    "notebooks/diff_in_diff_demo.ipynb": {
        "md": "# Difference-in-Differences — Notebook Demo\nSimple 2×2 DID with robust SE.",
        "src": "econometrics/diff_in_diff/did_basic.py",
    },
}

def build():
    Path("notebooks").mkdir(exist_ok=True)
    for nb_path, spec in MAP.items():
        md = spec["md"]
        code = Path(spec["src"]).read_text() if Path(spec["src"]).exists() else "print('Source missing')"
        nb = nbf.v4.new_notebook()
        nb.cells.append(nbf.v4.new_markdown_cell(md))
        nb.cells.append(nbf.v4.new_code_cell(code))
        nbf.write(nb, nb_path)
        print("written ->", nb_path)

if __name__ == "__main__":
    build()
