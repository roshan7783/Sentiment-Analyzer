import pandas as pd
import joblib
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from utils.text_cleaner import clean_text

# Load dataset (CSV must have 'text' and 'label')
data = pd.read_csv("sentiment_data.csv")

data['text'] = data['text'].apply(clean_text)

X = data['text']
y = data['label']  # 0=Negative, 1=Neutral, 2=Positive

vectorizer = TfidfVectorizer(max_features=5000)
X_vec = vectorizer.fit_transform(X)

model = LogisticRegression(max_iter=200)
model.fit(X_vec, y)

# SAVE MODEL FILES
joblib.dump(model, "models/tfidf_model.pkl")
joblib.dump(vectorizer, "models/vectorizer.pkl")

print("✅ Model trained & saved")