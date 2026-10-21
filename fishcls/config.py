from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data" / "Fish_Dataset"
CACHE_DIR = ROOT / "data" / "processed"
SPLITS_DIR = ROOT / "splits"
CONFIGS_DIR = ROOT / "configs"
RUNS_DIR = ROOT / "runs"
FIGURES_DIR = ROOT / "report" / "figures"

SEED = 42
IMAGE_EXTENSIONS = (".png", ".jpg", ".jpeg")
