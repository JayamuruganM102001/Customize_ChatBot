import pandas as pd
import spacy
import pickle
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import SVC

# Load dataset
df = pd.read_csv("training_data.csv")

# Load NLP model
nlp = spacy.load("en_core_web_sm")

# Preprocess text
def preprocess(text):
    doc = nlp(text.lower())
    return " ".join([token.lemma_ for token in doc if not token.is_punct and not token.is_stop])

df["processed_question"] = df["question"].apply(preprocess)

# Convert text to TF-IDF features
vectorizer = TfidfVectorizer()
X = vectorizer.fit_transform(df["processed_question"])
y = df["answer"]

# Train model
model = SVC(kernel="linear")
model.fit(X, y)

# Save model & vectorizer
with open("chatbot_model.pkl", "wb") as model_file:
    pickle.dump(model, model_file)

with open("vectorizer.pkl", "wb") as vectorizer_file:
    pickle.dump(vectorizer, vectorizer_file)

print("✅ Model training complete!")
