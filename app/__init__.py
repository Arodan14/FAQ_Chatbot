from flask import Flask, jsonify, render_template, request

from app.nlp.intent_recognizer import recognize_intent
import random


app = Flask(__name__)

WELCOME_MESSAGES = [
    "Welcome! Ask me a frequently asked question about Beykoz University.",
    "Hello! I can help with popular asked questions from the university FAQ.",
    "FAQ ready. Try a common student question or ask me to list the available FAQs.",
    "Need a quick answer? I match your message against the frequently asked questions dataset.",
    "Popular university questions start here. Ask me about registration, exams, attendance, or advisors.",
]


@app.route("/")
def home():
    return render_template(
        "chatbot.html",
        title="FAQ BOT | HOME",
        welcome_messages=WELCOME_MESSAGES,
        default_welcome_message=random.choice(WELCOME_MESSAGES),
        preset_prompt=request.args.get("prompt", ""),
        autorun_prompt=request.args.get("autorun", "0") == "1",
    )


@app.route("/about")
def about():
    return render_template("about.html", title="FAQ BOT | ABOUT")


@app.route("/settings")
def settings():
    return render_template("settings.html", title="FAQ BOT | SETTINGS")


@app.route("/help")
def help_page():
    sample_prompts = [
        "How do I register for courses?",
        "How can I add or drop a course?",
        "Who is my advisor?",
        "What is the attendance requirement?",
        "Where can I see exam dates?",
        "list faq questions",
    ]
    return render_template(
        "help.html",
        title="FAQ BOT | HELP",
        sample_prompts=sample_prompts,
    )


@app.route("/api/chat", methods=["POST"])
def chat():
    payload = request.get_json(silent=True) or {}
    user_input = payload.get("message", "")
    response = recognize_intent(user_input)
    return jsonify({"response": response})


if __name__ == "__main__":
    app.run(debug=True)
