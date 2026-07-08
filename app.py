from pathlib import Path
import runpy
import sys

ROOT = Path(__file__).resolve().parent
PROJECT_DIR = ROOT / "industrial_predictive_maintenance"

sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(PROJECT_DIR))

runpy.run_path(str(PROJECT_DIR / "app.py"), run_name="__main__")
