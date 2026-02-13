import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from PIL import Image
import tempfile
import cv2
import librosa
import speech_recognition as sr

st.set_page_config(
    page_title="Multimodal Sentiment AI",
    layout="wide"
)

st.title("🚀 Multimodal Sentiment & Emotion Dashboard")
st.markdown("Text • Face • Voice • CSV Analytics")

# =========================
# LOAD TEXT MODEL (Lazy)
# =========================
@st.cache_resource
def load_text_model():
    from transformers import pipeline
    return pipeline(
        "sentiment-analysis",
        model="distilbert-base-uncased-finetuned-sst-2-english"
    )

# =========================
# LOAD FACE MODEL (Lazy)
# =========================
@st.cache_resource
def load_fer_model():
    try:
        from fer.fer import FER
    except:
        from fer import FER
    return FER(mtcnn=True)

# =========================
# SIDEBAR NAVIGATION
# =========================
menu = st.sidebar.selectbox(
    "Select Analysis Type",
    [
        "📝 Live Text Sentiment",
        "🎤 Live Voice Sentiment",
        "📷 Live Face Emotion",
        "📊 CSV Dashboard"
    ]
)

# =========================
# TEXT SENTIMENT
# =========================
if menu == "📝 Live Text Sentiment":
    st.header("📝 Text Sentiment Analysis")

    text_input = st.text_area("Enter text for sentiment analysis")

    if st.button("Analyze Text"):
        if text_input.strip() == "":
            st.warning("Please enter text.")
        else:
            model = load_text_model()
            result = model(text_input)[0]

            st.success(f"Sentiment: {result['label']}")
            st.info(f"Confidence: {round(result['score']*100,2)}%")

# =========================
# VOICE SENTIMENT
# =========================
elif menu == "🎤 Live Voice Sentiment":
    st.header("🎤 Voice Sentiment Analysis")

    audio_file = st.file_uploader("Upload WAV file", type=["wav"])

    if audio_file:
        with tempfile.NamedTemporaryFile(delete=False) as tmp:
            tmp.write(audio_file.read())
            y, sr_rate = librosa.load(tmp.name)

            duration = len(y) / sr_rate
            st.success(f"Audio Duration: {round(duration,2)} seconds")

            recognizer = sr.Recognizer()
            with sr.AudioFile(tmp.name) as source:
                audio_data = recognizer.record(source)

                try:
                    text = recognizer.recognize_google(audio_data)
                    st.write("Transcribed Text:", text)

                    model = load_text_model()
                    result = model(text)[0]

                    st.info(f"Voice Sentiment: {result['label']}")
                    st.write(f"Confidence: {round(result['score']*100,2)}%")

                except:
                    st.error("Speech recognition failed.")

# =========================
# FACE EMOTION
# =========================
elif menu == "📷 Live Face Emotion":
    st.header("📷 Face Emotion Detection")

    image_file = st.file_uploader("Upload image", type=["jpg", "png", "jpeg"])

    if image_file:
        image = Image.open(image_file)
        st.image(image, caption="Uploaded Image", use_column_width=True)

        img_array = np.array(image)

        detector = load_fer_model()
        emotions = detector.detect_emotions(img_array)

        if emotions:
            st.json(emotions)
        else:
            st.warning("No face detected.")

# =========================
# CSV DASHBOARD
# =========================
elif menu == "📊 CSV Dashboard":
    st.header("📊 Customer Sentiment Dashboard")

    csv_file = st.file_uploader("Upload CSV file", type=["csv"])

    if csv_file:
        try:
            df = pd.read_csv(csv_file)

            st.write("Detected Columns:", df.columns.tolist())

            # Auto detect text column
            possible_cols = ["review_text", "review", "text", "feedback", "comments"]

            text_col = None
            for col in df.columns:
                if col.lower().strip() in possible_cols:
                    text_col = col
                    break

            if text_col is None:
                text_col = st.selectbox("Select text column manually", df.columns)

            df[text_col] = df[text_col].astype(str)

            model = load_text_model()

            sample_df = df.sample(min(500, len(df)))  # prevent memory crash
            predictions = model(sample_df[text_col].tolist())

            sample_df["Sentiment"] = [p["label"] for p in predictions]

            sentiment_counts = sample_df["Sentiment"].value_counts().reset_index()
            sentiment_counts.columns = ["Sentiment", "Count"]

            fig = px.pie(
                sentiment_counts,
                values="Count",
                names="Sentiment",
                title="Sentiment Distribution"
            )

            st.plotly_chart(fig, use_container_width=True)

        except Exception as e:
            st.error(f"Error processing CSV file: {str(e)}")

st.markdown("---")
st.markdown("🔥 AICTE IBM Multimodal AI Project | Built with Streamlit")
