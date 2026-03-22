from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path
import re

from scipy.sparse import hstack
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


APP_DIR = Path(__file__).resolve().parents[1]
DATA_PATH = APP_DIR / "processed_faqs.csv"
FALLBACK_DATA_PATH = APP_DIR / "BeykozUniFAQs.csv"
CALENDAR_FAQ_PATH = APP_DIR / "academic_calendar_faqs.csv"
UNKNOWN_RESPONSE = (
    "I could not find a confident answer for that question. "
    "Try rephrasing it or ask me to list the available FAQ questions."
)
EMPTY_RESPONSE = "Please type a question so I can help."

GREETING_KEYWORDS = {
    "hello",
    "hi",
    "hey",
    "good morning",
    "good afternoon",
    "good evening",
}

LIST_PATTERNS = (
    "list questions",
    "list faq",
    "show questions",
    "show faq",
    "all questions",
    "available questions",
    "what can you answer",
    "what can you help with",
)

STOPWORDS = {
    "a",
    "an",
    "and",
    "are",
    "can",
    "do",
    "for",
    "how",
    "i",
    "if",
    "in",
    "is",
    "it",
    "me",
    "my",
    "of",
    "on",
    "or",
    "the",
    "to",
    "what",
    "when",
    "where",
    "who",
    "why",
    "you",
    "your",
}

QUESTION_TYPE_KEYWORDS = {"how", "when", "where", "what", "why", "can", "do", "does", "is", "are"}


def _clean_text(value: object) -> str:
    text = "" if value is None else str(value)
    replacements = {
        "\u00a0": " ",
        "\u00c2": "",
        "\u00e2\u20ac\u2122": "'",
        "\u00e2\u20ac\u0153": '"',
        "\u00e2\u20ac\u009d": '"',
        "\u00e2\u20ac\u201c": "-",
    }
    for source, target in replacements.items():
        text = text.replace(source, target)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def _normalize(text: str) -> str:
    cleaned = _clean_text(text).lower()
    cleaned = re.sub(r"[^a-z0-9\s]", " ", cleaned)
    return re.sub(r"\s+", " ", cleaned).strip()


def _normalize_query_text(text: str) -> str:
    normalized = _normalize(text)
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
        "post graduate programs": "postgraduate programs",
        "post graduate program": "postgraduate program",
        "post graduate registration": "postgraduate registration",
    }
    for source, target in replacements.items():
        normalized = normalized.replace(source, target)
    return re.sub(r"\s+", " ", normalized).strip()


def _clean_question_text(question: str) -> str:
    cleaned_question = _clean_text(question)
    return re.sub(r"^\d+\.\s*", "", cleaned_question)


def _tokenize(text: str) -> list[str]:
    return [
        token
        for token in _stem_tokens(_normalize_query_text(text).split())
        if len(token) > 2 and token not in STOPWORDS
    ]


def _stem_tokens(tokens: list[str]) -> list[str]:
    stems: list[str] = []
    for token in tokens:
        stem = token
        for suffix in ("ation", "ing", "ment", "ity", "ies", "ied", "ed", "es", "s"):
            if len(stem) > 5 and stem.endswith(suffix):
                if suffix == "ies":
                    stem = stem[:-3] + "y"
                elif suffix == "ied":
                    stem = stem[:-3] + "y"
                else:
                    stem = stem[: -len(suffix)]
                break
        stems.append(stem)
    return stems


def _join_tokens(text: str) -> str:
    return " ".join(_tokenize(text))


@dataclass(frozen=True)
class FAQEntry:
    question: str
    intent: str
    answer: str
    search_text_override: str = ""
    source: str = "faq"
    topic: str = ""
    semester: str = ""

    @property
    def search_text(self) -> str:
        if self.search_text_override:
            return self.search_text_override
        normalized_question = _join_tokens(self.question)
        normalized_intent = _join_tokens(self.intent)
        return " ".join(
            part for part in (normalized_question, normalized_intent, normalized_question) if part
        )

    @property
    def question_type(self) -> str:
        tokens = _normalize_query_text(self.question).split()
        return tokens[0] if tokens and tokens[0] in QUESTION_TYPE_KEYWORDS else ""


class FAQService:
    def __init__(self, csv_path: Path = DATA_PATH) -> None:
        self.csv_path = csv_path if csv_path.exists() else FALLBACK_DATA_PATH
        self.entries = self._load_entries()
        self._fit_vectorizers()

    def _load_entries(self) -> list[FAQEntry]:
        cleaned_entries: list[FAQEntry] = []
        seen_questions: set[tuple[str, str]] = set()

        for path in [self.csv_path, CALENDAR_FAQ_PATH]:
            if not path.exists():
                continue

            with path.open("r", encoding="utf-8", errors="replace", newline="") as csv_file:
                reader = csv.DictReader(csv_file)
                expected_columns = {"Question", "Intent", "Answer"}
                fieldnames = set(reader.fieldnames or [])
                if not expected_columns.issubset(fieldnames):
                    missing = ", ".join(sorted(expected_columns - fieldnames))
                    raise ValueError(f"CSV is missing required columns: {missing}")

                for row in reader:
                    question = _clean_question_text(row.get("Question"))
                    intent = _clean_text(row.get("Intent"))
                    answer = _clean_text(row.get("Answer"))
                    search_text = _clean_text(row.get("SearchText"))
                    source = _clean_text(row.get("Source")) or "faq"
                    topic = _clean_text(row.get("Topic"))
                    semester = _clean_text(row.get("Semester"))

                    if not question or not intent or not answer:
                        continue

                    dedupe_key = (_normalize(question), source)
                    if dedupe_key in seen_questions:
                        continue

                    seen_questions.add(dedupe_key)
                    cleaned_entries.append(
                        FAQEntry(
                            question=question,
                            intent=intent,
                            answer=answer,
                            search_text_override=search_text,
                            source=source,
                            topic=topic,
                            semester=semester,
                        )
                    )

        if not cleaned_entries:
            raise ValueError("No valid FAQ entries were found in the CSV files.")

        return cleaned_entries

    def _legacy_load_entries(self) -> list[FAQEntry]:
        with self.csv_path.open("r", encoding="utf-8", errors="replace", newline="") as csv_file:
            reader = csv.DictReader(csv_file)
            expected_columns = {"Question", "Intent", "Answer"}
            fieldnames = set(reader.fieldnames or [])
            if not expected_columns.issubset(fieldnames):
                missing = ", ".join(sorted(expected_columns - fieldnames))
                raise ValueError(f"CSV is missing required columns: {missing}")

            cleaned_entries: list[FAQEntry] = []
            seen_questions: set[str] = set()

            for row in reader:
                question = _clean_question_text(row.get("Question"))
                intent = _clean_text(row.get("Intent"))
                answer = _clean_text(row.get("Answer"))
                search_text = _clean_text(row.get("SearchText"))

                if not question or not intent or not answer:
                    continue

                normalized_question = _normalize(question)
                if normalized_question in seen_questions:
                    continue

                seen_questions.add(normalized_question)
                cleaned_entries.append(
                    FAQEntry(
                        question=question,
                        intent=intent,
                        answer=answer,
                        search_text_override=search_text,
                    )
                )

        if not cleaned_entries:
            raise ValueError("No valid FAQ entries were found in the CSV file.")

        return cleaned_entries

    def _fit_vectorizers(self) -> None:
        corpus = [entry.search_text for entry in self.entries]

        self.word_vectorizer = TfidfVectorizer(
            analyzer="word",
            ngram_range=(1, 2),
            lowercase=False,
        )
        self.char_vectorizer = TfidfVectorizer(
            analyzer="char_wb",
            ngram_range=(3, 5),
            lowercase=False,
        )

        word_matrix = self.word_vectorizer.fit_transform(corpus)
        char_matrix = self.char_vectorizer.fit_transform(corpus)
        self.document_matrix = hstack([word_matrix, char_matrix]).tocsr()

    def _is_greeting(self, user_input: str) -> bool:
        normalized = _normalize(user_input)
        if not normalized:
            return False
        return normalized in GREETING_KEYWORDS

    def _wants_question_list(self, user_input: str) -> bool:
        normalized = _normalize(user_input)
        if not normalized:
            return False
        return any(pattern in normalized for pattern in LIST_PATTERNS)

    def _score_entries(self, user_input: str) -> list[tuple[float, FAQEntry]]:
        query_text = _join_tokens(user_input)
        if not query_text:
            return []

        word_query = self.word_vectorizer.transform([query_text])
        char_query = self.char_vectorizer.transform([query_text])
        query_matrix = hstack([word_query, char_query]).tocsr()

        similarities = cosine_similarity(query_matrix, self.document_matrix).flatten()
        query_type = self._question_type(user_input)
        ranked_entries = [
            (
                self._rerank_score(
                    float(similarities[index]),
                    query_type,
                    user_input,
                    self.entries[index],
                ),
                self.entries[index],
            )
            for index in range(len(self.entries))
        ]
        ranked_entries.sort(key=lambda item: item[0], reverse=True)
        return ranked_entries

    def _question_type(self, user_input: str) -> str:
        tokens = _normalize_query_text(user_input).split()
        return tokens[0] if tokens and tokens[0] in QUESTION_TYPE_KEYWORDS else ""

    def _rerank_score(
        self,
        base_score: float,
        query_type: str,
        user_input: str,
        entry: FAQEntry,
    ) -> float:
        score = base_score

        if query_type and query_type == entry.question_type:
            score += 0.05

        normalized_input = _normalize_query_text(user_input)
        if "fall semester" in normalized_input and entry.semester == "Fall Semester":
            score += 0.12
        if "spring semester" in normalized_input and entry.semester == "Spring Semester":
            score += 0.12
        if "summer term" in normalized_input and entry.semester == "Summer Term":
            score += 0.12
        if "public holidays" in normalized_input and entry.semester == "Public Holidays":
            score += 0.12
        if query_type == "when" and any(token in entry.search_text for token in ("time", "date", "schedule")):
            score += 0.05
        if query_type == "how" and any(token in entry.search_text for token in ("how", "procedure", "process", "apply")):
            score += 0.04
        if query_type == "can" and any(token in entry.search_text for token in ("can", "possibility", "availability")):
            score += 0.03
        if "grade" in normalized_input and "grade" in _normalize_query_text(entry.question):
            score += 0.05
        if "grade" in normalized_input and "grade" not in entry.search_text:
            score -= 0.08
        if "attendance" in normalized_input and "attendance" in entry.search_text:
            score += 0.05
        if "attendance" in normalized_input and "attendance" not in entry.search_text:
            score -= 0.06
        if "attendance" in normalized_input and "midterm" in entry.search_text:
            score -= 0.05
        if "postgraduate" in normalized_input and "postgraduate" in entry.search_text:
            score += 0.04

        return score

    def list_questions(self) -> str:
        lines = [
            f"{index}. {entry.question}"
            for index, entry in enumerate(self.entries, start=1)
            if entry.source != "calendar"
        ]
        return "Here are the FAQ questions I can answer:\n" + "\n".join(lines)

    def get_best_match(self, user_input: str) -> FAQEntry | None:
        ranked_entries = self._score_entries(user_input)
        if not ranked_entries:
            return None

        best_score, best_entry = ranked_entries[0]
        if best_score < 0.24:
            return None

        return best_entry

    def answer(self, user_input: str) -> str:
        cleaned_input = _clean_text(user_input)
        if not cleaned_input:
            return EMPTY_RESPONSE

        if self._wants_question_list(cleaned_input):
            return self.list_questions()

        if self._is_greeting(cleaned_input):
            return "Hello! Ask me a Beykoz University FAQ and I'll do my best to help."

        best_entry = self.get_best_match(cleaned_input)
        if best_entry is None:
            return UNKNOWN_RESPONSE

        semester_terms = {"fall", "spring", "summer"}
        normalized_input = _normalize_query_text(cleaned_input)
        if best_entry.source == "calendar" and not any(term in normalized_input for term in semester_terms):
            sibling_entries = [
                entry
                for entry in self.entries
                if entry.source == "calendar"
                and entry.topic == best_entry.topic
                and entry.semester
                and entry.semester != best_entry.semester
            ]
            if sibling_entries:
                return (
                    f"I found academic calendar information for {best_entry.topic.lower()}, "
                    "but it is time-sensitive. Please specify the semester, such as Fall, Spring, or Summer."
                )

        return best_entry.answer
