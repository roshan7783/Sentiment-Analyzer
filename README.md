# Sentiment Analysis Project

This project is a Sentiment Analysis and Emotion Detection application built with Streamlit. It allows users to analyze text from various sources like raw text, YouTube comments, and audio files. The application uses two different models for sentiment analysis: TF-IDF and BERT.

## Features

- **Text Analysis**: Analyze the sentiment and emotion of raw text.
- **Voice Analysis**: Upload a `.wav` file to get the sentiment and emotion of the transcribed text.
- **YouTube Analysis**: Analyze the sentiment of comments from a YouTube video.
- **CSV Bulk Upload**: Upload a CSV file with a 'text' column to get the sentiment and emotion for each row.
- **User Authentication**: Users can sign up and log in to use the application.
- **Model Selection**: Users can choose between two different models for sentiment analysis: TF-IDF and BERT.

## Project Structure

- `app.py`: The main Streamlit application file.
- `auth.py`: Handles user authentication (login and signup).
- `model.py`: Contains the TF-IDF sentiment analysis model.
- `bert_model.py`: Contains the BERT sentiment analysis model.
- `train_model.py`: Script to train the TF-IDF model.
- `requirements.txt`: A list of the Python dependencies for this project.
- `sentiment_data.csv`: The dataset used to train the TF-IDF model.
- `users.csv`: Stores user credentials.
- `utils/`: Contains utility functions for audio processing, emotion detection, text cleaning, and YouTube comment fetching.
- `models/`: Contains the pre-trained TF-IDF model and vectorizer.
- `.streamlit/config.toml`: Streamlit configuration file.

## Dependencies

The dependencies for this project are listed in the `requirements.txt` file. You can install them using pip:

```bash
pip install -r requirements.txt
```

## Setup and Installation

1. **Clone the repository:**

```bash
git clone https://github.com/your-username/sentiment-analysis-project.git
cd sentiment-analysis-project
```

2. **Install the dependencies:**

```bash
pip install -r requirements.txt
```

3. **Download NLTK data:**

The application uses the NLTK library for text processing. You will need to download the required NLTK data. Run the following command in your terminal:

```bash
python -m nltk.downloader stopwords
```

## How to Run the Project

To run the Streamlit application, use the following command in your terminal:

```bash
streamlit run app.py
```

This will open the application in your web browser.

## Models

### TF-IDF

The TF-IDF (Term Frequency-Inverse Document Frequency) model is a simple and effective model for sentiment analysis. It is trained on the `sentiment_data.csv` dataset.

### BERT

The BERT (Bidirectional Encoder Representations from Transformers) model is a state-of-the-art model for natural language processing. It is a more powerful model than the TF-IDF model and can achieve higher accuracy. The model used in this project is a pre-trained BERT model from the `transformers` library.

## Authentication

The application has a simple user authentication system. You can sign up for a new account or log in with an existing account. The user data is stored in the `users.csv` file.# Sentiment-Analyzer
