"""
Run the full RPA duration-prediction pipeline end to end.

Usage (from the repository root):
    uv run python main.py

Executes the five pipeline scripts in order and stops immediately if any of
them fails. The Streamlit app is intentionally NOT launched here; run it as a
separate step once the pipeline has produced
data/processed/DB_RPA_Projects_V9.xlsx:

    uv run streamlit run src/app.py
"""
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
SRC = ROOT / "src"

# Pipeline scripts, in execution order.
PIPELINE = [
    "01_preprocess.py",
    "02_univariate.py",
    "03_bivariate.py",
    "04_modeling.py",
    "05_extra_figures.py",
]


def run_step(script_name):
    """Run a single pipeline script and abort the whole run if it fails."""
    script_path = SRC / script_name

    print("\n" + "=" * 60)
    print(f">>> Running {script_name}")
    print("=" * 60, flush=True)

    result = subprocess.run([sys.executable, str(script_path)], cwd=SRC)

    if result.returncode != 0:
        print(f"\n[FAILED] {script_name} exited with code "
              f"{result.returncode}. Pipeline stopped.")
        sys.exit(result.returncode)

    print(f"[OK] {script_name} finished.")


def main():
    print("Starting the RPA duration-prediction pipeline...")

    for script in PIPELINE:
        run_step(script)

    print("\n" + "=" * 60)
    print("[DONE] Pipeline complete.")
    print("Launch the app with:  uv run streamlit run src/app.py")
    print("=" * 60)


if __name__ == "__main__":
    main()