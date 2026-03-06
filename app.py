from flask import Flask, render_template, request, jsonify
import numpy as np
import re

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from faq_data import faqs

app = Flask(__name__)

# Simple preprocessing (NO NLTK)
def preprocess(text):
    text = text.lower()
    text = re.sub(r'[^a-z0-9 ]', '', text)
    return text

questions = list(faqs.keys())
processed_questions = [preprocess(q) for q in questions]

vectorizer = TfidfVectorizer()
question_vectors = vectorizer.fit_transform(processed_questions)

def get_response(user_input):
    user_input = preprocess(user_input)
    user_vector = vectorizer.transform([user_input])

    similarity = cosine_similarity(user_vector, question_vectors)
    best_match = np.argmax(similarity)

    if similarity[0][best_match] > 0.3:
        return faqs[questions[best_match]]
    else:
        return "Sorry, I couldn't understand your question."

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/chat", methods=["POST"])
def chat():
    data = request.get_json()
    user_message = data.get("message", "")
    bot_reply = get_response(user_message)
    return jsonify({"reply": bot_reply})

if __name__ == "__main__":
    app.run(debug=True)
