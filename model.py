import joblib
from utils.text_cleaner import clean_text

model = joblib.load('models/tfidf_model.pkl')
vectorizer = joblib.load('models/vectorizer.pkl')

labels = {0: 'Negative', 1: 'Neutral', 2: 'Positive'}

def predict_sentiment(text):
    text = clean_text(text)
    vec = vectorizer.transform([text])
    pred = model.predict(vec)[0]
    prob = model.predict_proba(vec)[0]
    return labels[pred], prob