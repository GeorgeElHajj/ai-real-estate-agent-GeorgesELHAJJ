import json
from datetime import datetime
from pathlib import Path
from typing import Any


LOG_PATH = Path("logs/prompt_tests.jsonl")


def log_prompt_result(record: dict[str, Any]) -> None:
    LOG_PATH.parent.mkdir(parents=True, exist_ok=True)

    payload = {
        "timestamp": datetime.utcnow().isoformat(),
        **record,
    }

    with open(LOG_PATH, "a", encoding="utf-8") as f:
        f.write(json.dumps(payload, ensure_ascii=False) + "\n")