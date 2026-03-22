# FAQ Chatbot

University FAQ chatbot built with Flask and a `scikit-learn` NLP retrieval pipeline. The app matches student questions against a curated Beykoz University FAQ dataset and returns the closest FAQ answer through a simple web chat interface.

This repository includes:

- a Flask web app
- FAQ and academic calendar data pipelines
- a retrieval-based NLP matching system
- evaluation scripts and generated datasets

## Project summary

The chatbot is designed to answer university-related questions such as:

- course registration
- add/drop periods
- advisors
- attendance and exams
- academic calendar dates

It uses retrieval-based NLP rather than a generative model, which keeps the system explainable, lightweight, and easy to run locally.

## Project structure

```text
app/
  __init__.py
  BeykozUniFAQs.csv
  raw_faqs.csv
  raw_intents.csv
  processed_faqs.csv
  processed_intents.csv
  academic_calendar_events.csv
  academic_calendar_faqs.csv
  eval_queries.csv
  nlp/
    faq_service.py
    intent_recognizer.py
  static/
  templates/
scripts/
  data_preprocess.py
  process_academic_calendar.py
  evaluate_retrieval.py
requirements.txt
README.md
```

## Requirements

- Python 3.10+
- `pip`

Install the dependencies inside a virtual environment:

```bash
pip install -r requirements.txt
```

## How to run the Flask project

From the project root:

### PowerShell

```powershell
$env:FLASK_APP = "app"
$env:FLASK_ENV = "development"
flask run
```

### Or run it directly with Python

```powershell
python -m flask --app app --debug run
```

After that, open `http://127.0.0.1:5000`.

## App features

- Randomized FAQ-themed welcome message on the chat page
- Help page with ready-to-run sample prompts
- Settings page with browser-saved preferences for:
  - random welcome message
  - Enter-to-send
  - auto-scroll behavior
- Support for prefilled prompts from the Help page
- Semester-aware academic calendar answers for exam and registration date questions

## Data pipeline

The project now separates source data, processed data, and evaluation data.

- `app/raw_faqs.csv`: source FAQ dataset
- `app/raw_intents.csv`: source intent-answer dataset
- `app/processed_faqs.csv`: cleaned FAQ dataset used by the chatbot
- `app/processed_intents.csv`: cleaned intent dataset
- `app/academic_calendar_events.csv`: structured academic calendar events extracted from a PDF
- `app/academic_calendar_faqs.csv`: calendar entries converted into FAQ-style knowledge for the chatbot
- `app/eval_queries.csv`: example user queries for retrieval evaluation

The chatbot prefers `processed_faqs.csv` automatically. If `app/academic_calendar_faqs.csv` exists, it is also loaded as an additional knowledge source.

### Preprocess the data

Run:

```powershell
python scripts/data_preprocess.py
```

This script:

- cleans whitespace and broken characters
- removes numbering prefixes from questions
- removes empty rows
- deduplicates questions
- standardizes intent names
- regenerates processed FAQ and intent CSV files

### Evaluate retrieval accuracy

Run:

```powershell
python scripts/evaluate_retrieval.py
```

This evaluates the chatbot on `app/eval_queries.csv` by comparing the predicted intent with the expected intent.

Current evaluation snapshot:

- evaluation queries: `30`
- correct predictions: `28`
- accuracy: `93.33%`

You do not need to rerun this evaluation every time unless:

- you change the retrieval logic
- you change the dataset
- you add more evaluation queries

If the project stays unchanged, the current evaluation result is fine to keep in the README.

### Process an academic calendar PDF

Run:

```powershell
python scripts/process_academic_calendar.py "C:\path\to\academic-calendar.pdf"
```

This script:

- extracts text from the academic calendar PDF
- identifies calendar events and dates
- creates a structured event file
- creates FAQ-style calendar entries that the chatbot can load

Because academic calendar information is time-sensitive, calendar-derived answers should be interpreted by exact semester and date. The chatbot will ask the user to specify the semester when a calendar topic could apply to multiple terms such as Fall, Spring, or Summer.

Examples:

- `when do final exams start`
- `when do final exams start in the spring semester`
- `when is course registration in the fall semester`

## How the chatbot works

1. The user sends a message from the chat UI.
2. Flask posts that message to `/api/chat`.
3. The FAQ service preprocesses the input by cleaning punctuation, casing, spacing, and common noisy characters from the dataset.
4. The service checks for empty input, greetings, and requests to list all FAQ questions.
5. Each FAQ entry is converted into searchable text built from the cleaned question and intent label.
6. The project uses `scikit-learn` `TfidfVectorizer` features at both word and character level.
7. Cosine similarity is used to compare the user query against all FAQ entries.
8. If the top match is confident enough, the related answer is returned.
9. For time-sensitive calendar questions, the chatbot may ask the user to specify the semester before answering.
10. Otherwise, the chatbot asks the user to rephrase or request the available FAQ list.

## NLP approach

The chatbot uses a lightweight retrieval-based NLP pipeline instead of a generative model.

- Preprocessing:
  - question cleaning
  - whitespace normalization
  - removal of numbering prefixes from FAQ questions
  - token filtering with a small stopword list
- Feature extraction:
  - word-level TF-IDF with unigrams and bigrams
  - character-level TF-IDF with character windows
- Similarity:
  - cosine similarity between the user query and all FAQ entries
- Retrieval:
  - return the answer for the highest-scoring FAQ when the confidence threshold is met

This approach is explainable, reproducible, and lightweight enough to run locally without a large model.

## Evaluation

This project is structured as a retrieval-based NLP system rather than a supervised train/test classifier.

- The FAQ CSV acts as the retrieval knowledge base.
- `app/eval_queries.csv` acts as a lightweight evaluation set.
- The evaluation script measures whether a user query retrieves the expected intent.

This keeps the project aligned with FAQ retrieval while still giving you a measurable NLP evaluation workflow.

## Example prompts

- `How do I register for courses?`
- `How can I add or drop a course?`
- `I cannot access the university systems`
- `list faq questions`

## Notes

- This chatbot is limited to the FAQ data stored in `app/BeykozUniFAQs.csv`.
- The runtime now prefers `app/processed_faqs.csv`, which is generated from the raw data pipeline.
- A strong next improvement would be adding more paraphrased question variants for each intent.


