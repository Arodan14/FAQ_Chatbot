from __future__ import annotations

import csv
from pathlib import Path
import re


PROJECT_ROOT = Path(__file__).resolve().parents[1]
APP_DIR = PROJECT_ROOT / "app"
RAW_FAQ_PATH = APP_DIR / "raw_faqs.csv"
RAW_INTENT_PATH = APP_DIR / "raw_intents.csv"
PROCESSED_FAQ_PATH = APP_DIR / "processed_faqs.csv"
PROCESSED_INTENT_PATH = APP_DIR / "processed_intents.csv"


def clean_text(value: object) -> str:
    text = "" if value is None else str(value)
    replacements = {
        "\u00a0": " ",
        "\u00c2": "",
        "\u2019": "'",
        "\u2018": "'",
        "\u201c": '"',
        "\u201d": '"',
        "\u2013": "-",
        "\u2014": "-",
        "\u00e2\u20ac\u2122": "'",
        "\u00e2\u20ac\u0153": '"',
        "\u00e2\u20ac\u009d": '"',
        "\u00e2\u20ac\u201c": "-",
    }
    for source, target in replacements.items():
        text = text.replace(source, target)
    return re.sub(r"\s+", " ", text.replace("\r", " ").replace("\n", " ")).strip()


def normalize_query_text(text: str) -> str:
    normalized = clean_text(text).lower()
    replacements = {
        "log in": "access",
        "login": "access",
        "log into": "access",
        "sign in": "access",
        "sign into": "access",
        "course selection": "registration",
        "register": "registration",
        "registering": "registration",
        "passing": "successful",
        "pass": "successful",
        "failed": "fail",
        "failing": "fail",
        "class": "course",
        "classes": "attendance",
        "missing too many classes": "attendance exceeded",
        "missed too many classes": "attendance exceeded",
        "post graduate": "postgraduate",
        "post-graduate": "postgraduate",
    }
    for source, target in replacements.items():
        normalized = normalized.replace(source, target)
    return re.sub(r"\s+", " ", normalized).strip()


def clean_question(value: object) -> str:
    question = clean_text(value)
    return re.sub(r"^\d+\.\s*", "", question)


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", errors="replace", newline="") as csv_file:
        return list(csv.DictReader(csv_file))


def write_csv(path: Path, fieldnames: list[str], rows: list[dict[str, str]]) -> None:
    with path.open("w", encoding="utf-8", newline="") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def preprocess_faq_rows(rows: list[dict[str, str]]) -> list[dict[str, str]]:
    cleaned_rows: list[dict[str, str]] = []
    seen_questions: set[str] = set()

    for row in rows:
        question = clean_question(row.get("Question"))
        intent = clean_text(row.get("Intent"))
        answer = clean_text(row.get("Answer"))

        if not question or not intent or not answer:
            continue

        dedupe_key = question.lower()
        if dedupe_key in seen_questions:
            continue

        seen_questions.add(dedupe_key)
        cleaned_rows.append(
            {
                "Question": question,
                "Intent": intent,
                "Answer": answer,
                "SearchText": normalize_query_text(" ".join([question, intent])),
            }
        )

    return cleaned_rows


def preprocess_intent_rows(
    faq_rows: list[dict[str, str]],
    intent_rows: list[dict[str, str]],
) -> list[dict[str, str]]:
    intent_map: dict[str, str] = {}

    for row in faq_rows:
        intent_map[row["Intent"]] = row["Answer"]

    for row in intent_rows:
        intent = clean_text(row.get("Intent"))
        answer = clean_text(row.get("Answer"))
        if intent and answer and intent not in intent_map:
            intent_map[intent] = answer

    return [
        {"Intent": intent, "Answer": answer}
        for intent, answer in sorted(intent_map.items(), key=lambda item: item[0].lower())
    ]


def main() -> None:
    faq_rows = read_csv(RAW_FAQ_PATH)
    intent_rows = read_csv(RAW_INTENT_PATH)

    cleaned_faq_rows = preprocess_faq_rows(faq_rows)
    cleaned_intent_rows = preprocess_intent_rows(cleaned_faq_rows, intent_rows)

    write_csv(
        PROCESSED_FAQ_PATH,
        ["Question", "Intent", "Answer", "SearchText"],
        cleaned_faq_rows,
    )
    write_csv(PROCESSED_INTENT_PATH, ["Intent", "Answer"], cleaned_intent_rows)

    print(f"Raw FAQ rows: {len(faq_rows)}")
    print(f"Processed FAQ rows: {len(cleaned_faq_rows)}")
    print(f"Processed intent rows: {len(cleaned_intent_rows)}")
    print(f"Wrote: {PROCESSED_FAQ_PATH}")
    print(f"Wrote: {PROCESSED_INTENT_PATH}")


if __name__ == "__main__":
    main()
