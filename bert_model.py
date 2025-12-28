from transformers import pipeline

# Load model once (important for performance)
bert_sentiment = pipeline(
    "text-classification",
    model="cardiffnlp/twitter-roberta-base-sentiment",
    tokenizer="cardiffnlp/twitter-roberta-base-sentiment",
    truncation=True,
    max_length=512
)

def bert_predict(text):
    # Safety check
    if not text or text.strip() == "":
        # Return a neutral sentiment and even probability distribution if text is empty
        return "Neutral", {"Negative": 1/3, "Neutral": 1/3, "Positive": 1/3}

    # Truncate manually (extra safety)
    text = text[:2000]

    results = bert_sentiment(
        text,
        truncation=True,
        max_length=512,
        return_all_scores=True # Get probabilities for all labels
    )

    # results is a list containing one list of dictionaries, e.g., [[{'label': 'LABEL_0', 'score': 0.9}, ...]]
    # We need to extract the actual scores for each label
    all_scores = results[0]

    # Map labels to sentiment names and create a dictionary of probabilities
    prob_dict = {}
    label_map = {
        "LABEL_0": "Negative",
        "LABEL_1": "Neutral",
        "LABEL_2": "Positive"
    }

    for res in all_scores:
        sentiment_label = label_map.get(res["label"])
        if sentiment_label:
            prob_dict[sentiment_label] = res["score"]
    
    # Determine the overall sentiment (the one with the highest score)
    # The pipeline's default output (results[0][0] if return_all_scores=False) is already the highest
    # If we sort all_scores, the first element will be the highest.
    # We need to find the label that corresponds to the highest score from prob_dict
    predicted_sentiment = max(prob_dict, key=prob_dict.get)


    return predicted_sentiment, prob_dict