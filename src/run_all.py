"""Run the whole pipeline:  python src/run_all.py [--fast]"""
import subprocess, sys
from pathlib import Path

here = Path(__file__).resolve().parent
for script in ("01_validate_data.py", "02_build_dataset.py", "03_train_evaluate.py", "04_figures.py"):
    print(f"\n{'=' * 70}\n>>> {script}\n{'=' * 70}", flush=True)
    code = subprocess.run([sys.executable, str(here / script)] + sys.argv[1:]).returncode
    if code != 0:
        sys.exit(f"\nStopped: {script} failed (exit {code}). Fix the message above and re-run.")
print("\nALL DONE - see the 'outputs' folder.")
