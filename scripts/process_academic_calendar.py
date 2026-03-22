from __future__ import annotations

import csv
from pathlib import Path
import re
import sys

from pypdf import PdfReader


PROJECT_ROOT = Path(__file__).resolve().parents[1]
APP_DIR = PROJECT_ROOT / "app"
EVENTS_OUTPUT_PATH = APP_DIR / "academic_calendar_events.csv"
FAQ_OUTPUT_PATH = APP_DIR / "academic_calendar_faqs.csv"

SEMESTER_HEADERS = {
    "WORKING SCHEDULE FALL SEMESTER": "Fall Semester",
    "WORKING SCHEDULE SPRING SEMESTER": "Spring Semester",
    "WORKING SCHEDULE SUMMER TERM": "Summer Term",
    "PUBLIC HOLIDAYS": "Public Holidays",
}

INVALID_EVENT_NAMES = {
    "MONDAY",
    "TUESDAY",
    "WEDNESDAY",
    "THURSDAY",
    "FRIDAY",
    "SATURDAY",
    "SUNDAY",
    "ABSENTEEISM",
}

DROP_EVENT_NAMES = {
    "Programs)",
    "THE FALL SEMESTER",
    "AVERAGE",
    "EDUCATION INSTITUTIONS WITHIN THE SCOPE OF RECOGNITION OF PRIOR LEARNING",
    "teaching methods, assessment/evaluation methods and ratios of the courses.",
    "Article-1)",
}

DROP_EVENT_CONTAINS = (
    "teaching methods, assessment/evaluation methods",
    "recognition of prior learning",
)

MONTH_PATTERN = (
    r"January|February|March|April|May|June|July|August|September|October|November|December"
)
DATE_PATTERN = re.compile(
    rf"(?P<date>(?:"
    rf"(?:{MONTH_PATTERN})\s*-\s*\d{{1,2}}\s+(?:{MONTH_PATTERN})\s+\d{{4}}|"
    rf"\d{{1,2}}\s+(?:{MONTH_PATTERN})\s*-\s*\d{{1,2}}\s+(?:{MONTH_PATTERN})\s+\d{{4}}|"
    rf"\d{{1,2}}\s*-\s*\d{{1,2}}\s+(?:{MONTH_PATTERN})\s+\d{{4}}|"
    rf"\d{{1,2}}\s+(?:{MONTH_PATTERN})\s+\d{{4}})"
    rf"(?:.*))"
)


def clean_text(value: object) -> str:
    text = "" if value is None else str(value)
    replacements = {
        "\u00a0": " ",
        "\u2019": "'",
        "\u2018": "'",
        "\u201c": '"',
        "\u201d": '"',
        "\u2013": "-",
        "\u2014": "-",
    }
    for source, target in replacements.items():
        text = text.replace(source, target)
    return re.sub(r"\s+", " ", text).strip()


def extract_lines(pdf_path: Path) -> list[str]:
    reader = PdfReader(str(pdf_path))
    lines: list[str] = []
    for page in reader.pages:
        text = page.extract_text() or ""
        lines.extend(clean_text(line) for line in text.splitlines())
    return [line for line in lines if line]


def is_noise(line: str) -> bool:
    return line.startswith("BEYKOZ UNIVERSITY") or line.startswith("2025-2026 ACADEMIC CALENDAR")


def normalize_event_name(text: str) -> str:
    text = clean_text(text)
    text = re.sub(r"\s+\d{1,2}$", "", text)
    return text


def normalize_semester(event_name: str, semester: str) -> str:
    upper_name = event_name.upper()
    if "(HOLIDAY)" in upper_name or upper_name in {
        "DEMOCRACY AND NATIONAL UNITY DAY",
        "NEW YEAR'S DAY (HOLIDAY)",
    }:
        return "Public Holidays"
    return semester


def is_invalid_event_name(event_name: str) -> bool:
    cleaned = clean_text(event_name)
    if not cleaned:
        return True
    if cleaned in DROP_EVENT_NAMES:
        return True
    if any(fragment.lower() in cleaned.lower() for fragment in DROP_EVENT_CONTAINS):
        return True
    if cleaned.upper() in INVALID_EVENT_NAMES:
        return True
    if re.fullmatch(r"[\d\s\-\/]+", cleaned):
        return True
    if len(cleaned) < 4:
        return True
    if cleaned.endswith(")") and len(cleaned) < 18:
        return True
    if cleaned.startswith("THE ") and len(cleaned.split()) <= 4:
        return True
    return False


def parse_events(lines: list[str]) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    seen: set[tuple[str, str, str]] = set()
    current_semester = ""
    current_row: dict[str, str] | None = None
    pending_prefix = ""

    for line in lines:
        if line in SEMESTER_HEADERS:
            current_semester = SEMESTER_HEADERS[line]
            current_row = None
            pending_prefix = ""
            continue

        if is_noise(line):
            continue

        matches = list(DATE_PATTERN.finditer(line))
        match = matches[0] if matches else None
        if match and current_semester:
            date_text = clean_text(match.group("date"))
            raw_event_name = normalize_event_name(line[: match.start("date")])
            if pending_prefix:
                if raw_event_name:
                    event_name = normalize_event_name(f"{pending_prefix} {raw_event_name}")
                else:
                    event_name = normalize_event_name(pending_prefix)
                pending_prefix = ""
            else:
                event_name = raw_event_name

            if not event_name and current_row is not None:
                current_row["details"] = clean_text(
                    " ".join([current_row["details"], line])
                )
                continue

            if is_invalid_event_name(event_name):
                current_row = None
                continue

            if not event_name:
                if current_row is not None:
                    current_row["details"] = clean_text(
                        " ".join([current_row["details"], line])
                    )
                continue

            row = {
                "semester": normalize_semester(event_name, current_semester),
                "event_name": event_name,
                "date_text": date_text,
                "details": "",
            }
            if is_invalid_event_name(row["event_name"]):
                current_row = None
                continue
            row_key = (row["semester"], row["event_name"], row["date_text"])
            if row_key not in seen:
                rows.append(row)
                seen.add(row_key)
                current_row = row
            else:
                current_row = None
            continue

        if current_row is not None and not pending_prefix:
            current_row["details"] = clean_text(" ".join([current_row["details"], line]))
        else:
            pending_prefix = clean_text(" ".join([pending_prefix, line]))

    return rows


def make_question(event_name: str, semester: str) -> str:
    return f"When is {event_name.lower()} in the {semester.lower()}?"


def make_answer(event_name: str, semester: str, date_text: str, details: str) -> str:
    answer = f"{event_name.title()} for the {semester} is scheduled for {date_text}."
    if details:
        answer += f" Additional details: {details}"
    return answer


def make_search_text(event_name: str, semester: str, date_text: str) -> str:
    normalized_event = clean_text(event_name).lower()
    normalized_semester = clean_text(semester).lower()
    return clean_text(
        f"{normalized_event} {normalized_semester} academic calendar date deadline schedule {date_text}"
    )


def write_csv(path: Path, fieldnames: list[str], rows: list[dict[str, str]]) -> None:
    with path.open("w", encoding="utf-8", newline="") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    if len(sys.argv) < 2:
        raise SystemExit("Usage: python scripts/process_academic_calendar.py <pdf_path>")

    pdf_path = Path(sys.argv[1])
    if not pdf_path.exists():
        raise SystemExit(f"PDF not found: {pdf_path}")

    lines = extract_lines(pdf_path)
    events = parse_events(lines)

    faq_rows = [
        {
            "Question": make_question(row["event_name"], row["semester"]),
            "Intent": f"Academic Calendar | {row['event_name']} | {row['semester']}",
            "Answer": make_answer(
                row["event_name"], row["semester"], row["date_text"], row["details"]
            ),
            "SearchText": make_search_text(
                row["event_name"], row["semester"], row["date_text"]
            ),
            "Source": "calendar",
            "Topic": row["event_name"],
            "Semester": row["semester"],
        }
        for row in events
    ]

    write_csv(
        EVENTS_OUTPUT_PATH,
        ["semester", "event_name", "date_text", "details"],
        events,
    )
    write_csv(
        FAQ_OUTPUT_PATH,
        ["Question", "Intent", "Answer", "SearchText", "Source", "Topic", "Semester"],
        faq_rows,
    )

    print(f"Parsed calendar events: {len(events)}")
    print(f"Wrote: {EVENTS_OUTPUT_PATH}")
    print(f"Wrote: {FAQ_OUTPUT_PATH}")


if __name__ == "__main__":
    main()
