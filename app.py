import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import re # Import the regex module
import altair as alt # Import altair

# Auth
from auth import login_page, signup_page, logout_button

# Models
from model import predict_sentiment
from bert_model import bert_predict

# Utils
from utils.audio_utils import speech_to_text, record_and_recognize_speech
from utils.youtube_utils import get_comments
from utils.emotion_utils import detect_emotion


# Helper function to check if a string is a YouTube URL
def is_youtube_url(url):
    youtube_regex = (
        r'(https?://)?(www\.)?'
        '(youtube|youtu|youtube-nocookie)\.(com|be)/'
        '(watch\?v=|embed/|v/|.+\?v=)?([^&=%\?]{11})')
    return re.match(youtube_regex, url)


# --------------------------------------------------
# PAGE CONFIG
# --------------------------------------------------
st.set_page_config(page_title="Sentiment Analyzer", layout="wide")


# --------------------------------------------------
# MAIN BRAND (CENTER)
# --------------------------------------------------
st.markdown("<div class='brand'>Sentiment Analyzer</div>", unsafe_allow_html=True)
st.markdown("<center><p>Advanced Sentiment & Emotion Analysis Platform</p></center>", unsafe_allow_html=True)


st.divider()



# --------------------------------------------------
# GLOBAL CSS (Peacock Blue Theme)
# --------------------------------------------------
st.markdown("""
<style>
/* Buttons */
.stButton > button {
    background-color: #F32816;
    color: white;
    border-radius: 8px;
    padding: 0.5em 1.2em;
    border: none;
}
.stButton > button:hover {
    background-color: #F32816;
    color: white;
}

/* Center brand title */
.brand {
    text-align: center;
    font-size: 42px;
    font-weight: 700;
    color: #F32816;
    margin-bottom: 10px;
}

/* Sidebar footer auth */
.sidebar-auth {
    position: fixed;
    bottom: 20px;
    width: 300px;
}

/* Netflix-like Description Style */
.netflix-description {
    text-align: center;
    font-size: 18px;
    color: #f0f2f6; /* Light gray for contrast */
    margin-top: 10px;
    margin-bottom: 20px;
    line-height: 1.5;
}

.netflix-description .red-text {
    color: #F32816; /* Brand red */
    font-weight: bold;
}
</style>
""", unsafe_allow_html=True)


# --------------------------------------------------
# SESSION STATE
# --------------------------------------------------
if "logged_in" not in st.session_state:
    st.session_state.logged_in = False


# --------------------------------------------------
# SIDEBAR AUTH (BOTTOM)
# --------------------------------------------------
with st.sidebar:
    st.markdown("### 🔐 Account")

    if not st.session_state.logged_in:
        auth_choice = st.radio("Select", ["Login", "Sign Up"])

        st.markdown("<div class='sidebar-auth'>", unsafe_allow_html=True)
        if auth_choice == "Login":
            login_page()
        else:
            signup_page()
        st.markdown("</div>", unsafe_allow_html=True)

        st.stop()
    else:
        st.success(f"Logged in as {st.session_state.user}")
        logout_button()



# --------------------------------------------------
# FEATURE MENU
# --------------------------------------------------
menu = st.sidebar.selectbox(
    "📌 Select Feature",
    ["Text Analysis", "Voice Analysis", "YouTube Analysis", "CSV Bulk Upload"]
)

model_type = st.sidebar.radio("🤖 Select Model", ["TF-IDF", "BERT"])


# --------------------------------------------------
# TEXT ANALYSIS
# --------------------------------------------------
if menu == "Text Analysis":
    text = st.text_area("✍️ Enter Text")

    if st.button("Analyze"):
        if not text.strip():
            st.warning("Please enter some text")
        elif is_youtube_url(text):
            st.subheader("YouTube Comment Sentiment Analysis")
            comments = get_comments(text)

            if not comments:
                st.info("No comments found for this video or comments could not be fetched.")
            else:
                total_probs = {'Negative': 0.0, 'Neutral': 0.0, 'Positive': 0.0}
                comment_count = 0

                progress_bar = st.progress(0)
                status_text = st.empty()

                for i, c in enumerate(comments):
                    if not c or len(c.strip()) < 5:
                        continue
                    
                    c_truncated = c[:2000] # Truncate comment for analysis

                    if model_type == "BERT":
                        _, prob_dict = bert_predict(c_truncated)
                    else: # TF-IDF
                        _, prob_array = predict_sentiment(c_truncated)
                        # Convert array to dict for consistency with BERT output
                        prob_dict = {
                            'Negative': prob_array[0],
                            'Neutral': prob_array[1],
                            'Positive': prob_array[2]
                        }
                    
                    for sentiment_label, prob_value in prob_dict.items():
                        total_probs[sentiment_label] += prob_value
                    comment_count += 1
                    
                    progress_bar.progress((i + 1) / len(comments))
                    status_text.text(f"Analyzing comment {i+1} of {len(comments)}")

                if comment_count > 0:
                    avg_probs = {k: v / comment_count for k, v in total_probs.items()}
                    
                    st.write(f"Analyzed **{comment_count}** comments.")
                    st.success(f"**Positive:** {avg_probs['Positive']:.2%}")
                    st.info(f"**Neutral:** {avg_probs['Neutral']:.2%}")
                    st.warning(f"**Negative:** {avg_probs['Negative']:.2%}")

                    # Optional: Display a bar chart for average probabilities
                    fig, ax = plt.subplots()
                    ax.bar(avg_probs.keys(), avg_probs.values(), color=['red', 'gray', 'green'])
                    plt.xticks(rotation=45, ha="right")
                    ax.set_ylabel("Average Probability")
                    ax.set_title("Average YouTube Comment Sentiment Distribution")
                    st.pyplot(fig)
                else:
                    st.info("No valid comments to analyze.")
        else:
            sentiment, prob_value = (
                bert_predict(text) if model_type == "BERT"
                else predict_sentiment(text)
            )

            # If TF-IDF, convert prob_value (array) to a dictionary for display consistency
            if model_type == "TF-IDF":
                # Assuming order [Negative, Neutral, Positive]
                prob_dict_display = {'Negative': prob_value[0], 'Neutral': prob_value[1], 'Positive': prob_value[2]}
                st.success(f"Sentiment: {sentiment}")
                st.write(f"**Positive:** {prob_dict_display['Positive']:.2%}")
                st.info(f"**Neutral:** {prob_dict_display['Neutral']:.2%}")
                st.warning(f"**Negative:** {prob_dict_display['Negative']:.2%}")
            else: # BERT already returns a dictionary
                st.success(f"Sentiment: {sentiment}")
                st.write(f"**Positive:** {prob_value['Positive']:.2%}")
                st.info(f"**Neutral:** {prob_value['Neutral']:.2%}")
                st.warning(f"**Negative:** {prob_value['Negative']:.2%}")

            emotion, emo_score = detect_emotion(text)
            st.write(f"Emotion: {emotion} ({emo_score:.2f})")


# --------------------------------------------------
# VOICE ANALYSIS
# --------------------------------------------------
elif menu == "Voice Analysis":
    audio = st.file_uploader("🎤 Upload WAV file", type=["wav"])

    if audio:
        text = speech_to_text(audio)
        st.write("Recognized Text:", text)

        sentiment, score = bert_predict(text)
        emotion, _ = detect_emotion(text)

        st.success(f"Sentiment: {sentiment}")
        st.warning(f"Emotion: {emotion}")


# --------------------------------------------------
# YOUTUBE ANALYSIS
# --------------------------------------------------
elif menu == "YouTube Analysis":
    url = st.text_input("📺 YouTube Video URL")

    if st.button("Analyze Comments"):
        comments = get_comments(url)
        results = {"Positive": 0, "Neutral": 0, "Negative": 0}

        for c in comments:
            if not c or len(c.strip()) < 5:
                continue
            c = c[:2000]
            s, _ = bert_predict(c) # Using BERT for consistency as it was previously
            results[s] += 1

        st.subheader("📊 Sentiment Distribution")

        if sum(results.values()) > 0:
            # Prepare data for st.bar_chart
            df_results = pd.DataFrame(results.items(), columns=['Sentiment', 'Count'])
            
            # Create Altair chart
            chart = alt.Chart(df_results).mark_bar().encode(
                x=alt.X('Sentiment', axis=alt.Axis(labelAngle=-360)), # Rotate labels
                y='Count'
            ).properties(
                title='YouTube Comment Sentiment Distribution'
            )
            st.altair_chart(chart, use_container_width=True)

            st.markdown(f"**Positive Comments:** {results['Positive']} ")
            st.markdown(f"**Neutral Comments:** {results['Neutral']} ")
            st.markdown(f"**Negative Comments:** {results['Negative']} ")
        else:
            st.info("No comments found or analyzed to display sentiment distribution.")


# --------------------------------------------------
# CSV BULK UPLOAD
# --------------------------------------------------
elif menu == "CSV Bulk Upload":
    file = st.file_uploader("📂 Upload CSV with 'text' column", type=["csv"])

    if file:
        df = pd.read_csv(file)

        if "text" not in df.columns:
            st.error("CSV must contain a 'text' column")
        else:
            df["Sentiment"] = df["text"].apply(
                lambda x: bert_predict(str(x)[:2000])[0]
            )
            df["Emotion"] = df["text"].apply(
                lambda x: detect_emotion(str(x))[0]
            )

            st.dataframe(df)