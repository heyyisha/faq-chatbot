from flask import Flask, render_template, request, jsonify
import json
import spacy
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

app = Flask(__name__)

# Load the FAQs
with open('faqs.json', 'r') as f:
    faqs = json.load(f)

# Load the English NLP model from spaCy
nlp = spacy.load('en_core_web_sm')

# Preprocess text: lowercase, remove stopwords and punctuation
def preprocess(text):
    doc = nlp(text.lower())
    tokens = [token.lemma_ for token in doc if not token.is_stop and not token.is_punct]
    return ' '.join(tokens)

# Preprocess all questions
questions = [q['question'] for q in faqs]
processed_questions = [preprocess(q) for q in questions]

# Initialize TF-IDF vectorizer
vectorizer = TfidfVectorizer()
question_vectors = vectorizer.fit_transform(processed_questions)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/ask', methods=['POST'])
def ask():
    user_question = request.json['question']
    
    # Preprocess the user's question
    processed_user_question = preprocess(user_question)
    user_vector = vectorizer.transform([processed_user_question])
    
    # Calculate cosine similarity between user question and FAQs
    similarities = cosine_similarity(user_vector, question_vectors)
    best_match_index = similarities.argmax()
    best_match_score = similarities[0, best_match_index]
    
    # Only return a match if the similarity score is above a threshold
    if best_match_score > 0.3:  # Adjust this threshold as needed
        best_answer = faqs[best_match_index]['answer']
        return jsonify({'answer': best_answer})
    else:
        return jsonify({'answer': "Sorry, I don't understand that question. Please try rephrasing or ask another question about Python."})

if __name__ == '__main__':
    app.run(debug=True)