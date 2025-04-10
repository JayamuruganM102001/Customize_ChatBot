import pickle
import spacy
from flask import Flask, request, jsonify
from flask_cors import CORS  # Import CORS to avoid cross-origin issues

# Load trained model & vectorizer
with open("chatbot_model.pkl", "rb") as model_file:
    model = pickle.load(model_file)

with open("vectorizer.pkl", "rb") as vectorizer_file:
    vectorizer = pickle.load(vectorizer_file)

nlp = spacy.load("en_core_web_sm")

# Flask API
app = Flask(__name__)
CORS(app)  # Enable CORS

@app.route("/chat", methods=["POST"])
def chat():
    try:
        data = request.json
        user_message = data.get("message", "").strip()

        if not user_message:
            return jsonify({"response": "Please enter a valid message!"})

        # Preprocess input
        doc = nlp(user_message.lower())
        processed_input = " ".join([token.lemma_ for token in doc if not token.is_punct and not token.is_stop])

        # Convert input to TF-IDF features
        input_vector = vectorizer.transform([processed_input])

        # Get chatbot response
        response = model.predict(input_vector)[0]

        return jsonify({"response": response})
    
    except Exception as e:
        return jsonify({"response": f"Error: {str(e)}"})

if __name__ == "__main__":
    app.run(port=5000, debug=True)
