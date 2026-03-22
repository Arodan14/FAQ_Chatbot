from __future__ import annotations

import csv
from pathlib import Path
import sys


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from app.nlp.faq_service import FAQService


EVAL_PATH = PROJECT_ROOT / "app" / "eval_queries.csv"


def main() -> None:
    service = FAQService()

    with EVAL_PATH.open("r", encoding="utf-8", newline="") as csv_file:
        rows = list(csv.DictReader(csv_file))

    total = 0
    correct = 0

    for row in rows:
        query = row.get("query", "").strip()
        expected_intent = row.get("expected_intent", "").strip()
        if not query or not expected_intent:
            continue

        total += 1
        match = service.get_best_match(query)
        predicted_intent = match.intent if match is not None else "unknown"
        is_correct = predicted_intent == expected_intent
        if is_correct:
            correct += 1

        status = "OK" if is_correct else "MISS"
        print(f"[{status}] query={query}")
        print(f"       expected={expected_intent}")
        print(f"       predicted={predicted_intent}")

    accuracy = (correct / total * 100) if total else 0.0
    print(f"\nAccuracy: {correct}/{total} ({accuracy:.2f}%)")


if __name__ == "__main__":
    main()
