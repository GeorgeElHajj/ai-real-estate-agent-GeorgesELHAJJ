from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent.parent

MODEL_ARTIFACT_PATH = BASE_DIR / "ml" / "artifacts" / "best_model.joblib"
TRAINING_STATS_PATH = BASE_DIR / "ml" / "artifacts" / "training_stats.json"
PROMPTS_DIR = BASE_DIR / "prompts"
LOGS_DIR = BASE_DIR / "logs"