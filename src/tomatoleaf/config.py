from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
DATASET_DIR = ROOT / "Dataset" / "Variant-a(Multiclass Classification)"
DATA_DIR = ROOT / "data"
EXPERIMENTS_DIR = ROOT / "experiments"
MODELS_DIR = EXPERIMENTS_DIR / "models"
RESULTS_DIR = EXPERIMENTS_DIR / "results"
FIGURES_DIR = ROOT / "reports" / "figures"

SEED = 42
OFFICIAL_SPLITS = ("train", "val", "test")
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

# Fixed class order used by every model, the API and the UI (alphabetical by dataset folder).
CLASSES = [
    {"index": 0, "key": "early_blight", "folder": "Early_blight", "name": "Early Blight"},
    {"index": 1, "key": "healthy", "folder": "Healthy", "name": "Healthy"},
    {"index": 2, "key": "late_blight", "folder": "Late_blight", "name": "Late Blight"},
    {"index": 3, "key": "leaf_miner", "folder": "Leaf Miner", "name": "Leaf Miner"},
    {"index": 4, "key": "magnesium_deficiency", "folder": "Magnesium Deficiency", "name": "Magnesium Deficiency"},
    {"index": 5, "key": "nitrogen_deficiency", "folder": "Nitrogen Deficiency", "name": "Nitrogen Deficiency"},
    {"index": 6, "key": "potassium_deficiency", "folder": "Pottassium Deficiency", "name": "Potassium Deficiency"},
    {"index": 7, "key": "spotted_wilt_virus", "folder": "Spotted Wilt Virus", "name": "Spotted Wilt Virus"},
]
NUM_CLASSES = len(CLASSES)
CLASS_KEYS = [c["key"] for c in CLASSES]
CLASS_NAMES = [c["name"] for c in CLASSES]

# Every dataset image is 256x256; uploads are mapped to the same geometry before any model sees them.
CANONICAL_SIZE = 256
MODEL_INPUT_SIZE = 224
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)
