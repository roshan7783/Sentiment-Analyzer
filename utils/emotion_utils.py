from transformers import pipeline

emotion_classifier = pipeline(
    "text-classification",
    model="j-hartmann/emotion-english-distilroberta-base",
    return_all_scores=True
)

def detect_emotion(text):
    emotions = emotion_classifier(text)[0]
    top = max(emotions, key=lambda x: x['score'])
    return top['label'], top['score']