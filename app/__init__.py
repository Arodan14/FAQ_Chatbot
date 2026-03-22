from flask import Flask, jsonify, render_template, request

from app.nlp.intent_recognizer import recognize_intent
import random


app = Flask(__name__)

WELCOME_MESSAGES = [
    "Welcome to FAQ_BOT. Ask me a frequently asked question about Beykoz University.",
    "Hello from FAQ_BOT. I can help with popular university questions.",
    "FAQ_BOT is ready. Try a common student question or ask me to list the available questions.",
    "Need a quick answer? FAQ_BOT matches your message against the FAQ dataset.",
    "FAQ_BOT is here to help with registration, exams, attendance, advisors, and more.",
]


@app.route("/")
def home():
    return render_template(
        "chatbot.html",
        title="FAQ_BOT | HOME",
        welcome_messages=WELCOME_MESSAGES,
        default_welcome_message=random.choice(WELCOME_MESSAGES),
        preset_prompt=request.args.get("prompt", ""),
        autorun_prompt=request.args.get("autorun", "0") == "1",
    )


@app.route("/about")
def about():
    return render_template("about.html", title="FAQ_BOT | ABOUT")


@app.route("/settings")
def settings():
    return render_template("settings.html", title="FAQ_BOT | SETTINGS")


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
        title="FAQ_BOT | HELP",
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
