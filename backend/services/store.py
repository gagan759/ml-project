"""Read-only JSON content served by the API: the platform analytics and the disease library."""
from pathlib import Path

from tomatoleaf.config import EXPERIMENTS_DIR
from tomatoleaf.utils import load_json

DISEASE_INFO = Path(__file__).resolve().parents[1] / "data" / "disease_info.json"
ANALYTICS = EXPERIMENTS_DIR / "platform.json"  # written by scripts/27_export_platform_data.py


class DataStore:
    def __init__(self, analytics_path: Path = ANALYTICS, disease_path: Path = DISEASE_INFO):
        self.analytics = load_json(analytics_path) if analytics_path.exists() else None
        self.diseases = load_json(disease_path) if disease_path.exists() else None
