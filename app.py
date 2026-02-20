import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import tempfile
import os
import cv2
import librosa
import speech_recognition as sr
from fer import FER
from transformers import pipeline
import google.generativeai as genai
import datetime

# ================= CONFIG =================
st.set_page_config(page_title="AI Customer Vision Pro", layout="wide")

st.title("🌟 AI Customer Vision Pro")
st.markdown("Multimodal Customer Sentiment Intelligence Platform")

# ================= GEMINI =================
if "GEMINI_API_KEY" in st.secrets:
    genai.configure(api_key=st.secrets["GEMINI_API_KEY"])
    gemini = genai.GenerativeModel("gemini-pro")
else:
    gemini = None
    st.warning("Add GEMINI_API_KEY in Secrets")

# ================= MODELS =================
@st.cache_resource
def load_sentiment():
    return pipeline("sentiment-analysis")

sentiment_model = load_sentiment()

# ================= FUNCTIONS =================

def analyze_text(text):
    result = sentiment_model(text)[0]
    return result["label"], result["score"]


def analyze_voice(audio_file):
    recognizer = sr.Recognizer()

    with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp:
        tmp.write(audio_file.read())
        path = tmp.name

    with sr.AudioFile(path) as source:
        audio = recognizer.record(source)

    try:
        text = recognizer.recognize_google(audio)
    except:
        text = "Could not understand"

    sentiment, score = analyze_text(text)
    return text, sentiment, score


# Replace your current analyze_video function with this:
def analyze_video(video_file):
    from transformers import pipeline
    from PIL import Image
    
    # Load a robust image emotion classifier
    classifier = pipeline("image-classification", model="dima806/facial_emotions_image_detection")
    
    with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as tmp:
        tmp.write(video_file.read())
        path = tmp.name

    cap = cv2.VideoCapture(path)
    emotions = []
    
    # Sample 5-10 frames for processing speed
    for i in range(25):
        ret, frame = cap.read()
        if not ret: break
        if i % 5 == 0:
            img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            pil_img = Image.fromarray(img)
            res = classifier(pil_img)
            emotions.append(res[0]['label'])
            
    cap.release()
    
    if emotions:
        return max(set(emotions), key=emotions.count)
    return "Neutral"


def gemini_generate(prompt):
    if gemini is None:
        return "Gemini not configured"
    return gemini.generate_content(prompt).text


def fusion_score(text_sent, voice_sent, video_emotion):

    score_map = {
        "POSITIVE": 1,
        "NEGATIVE": -1,
        "NEUTRAL": 0,
        "happy": 1,
        "sad": -1,
        "angry": -1,
        "neutral": 0
    }

    scores = []

    if text_sent:
        scores.append(score_map.get(text_sent.upper(), 0))

    if voice_sent:
        scores.append(score_map.get(voice_sent.upper(), 0))

    if video_emotion:
        scores.append(score_map.get(video_emotion.lower(), 0))

    if scores:
        final = np.mean(scores)
        if final > 0.2:
            return "Positive"
        elif final < -0.2:
            return "Negative"
        else:
            return "Neutral"

    return "Not enough data"


# ================= SESSION =================
if "history" not in st.session_state:
    st.session_state.history = []

# ================= INPUT =================
st.sidebar.header("Upload CSV Data")
csv_file = st.sidebar.file_uploader("Upload CSV", type=["csv"])

text_input = st.text_area("Enter Text")

audio_file = st.file_uploader("Upload Audio (wav)", type=["wav"])

video_file = st.file_uploader("Upload Video (mp4)", type=["mp4"])

# ================= TABS =================
tab1, tab2, tab3 = st.tabs([
    "📊 Dashboard",
    "🧠 Gemini AI",
    "📈 Fusion Intelligence"
])

# ================= DASHBOARD =================
with tab1:

    if csv_file:
        df = pd.read_csv(csv_file)

        if "review" in df.columns:
            df["sentiment"] = df["review"].apply(lambda x: analyze_text(str(x))[0])

            st.dataframe(df.head())

            fig = px.histogram(df, x="sentiment", color="sentiment")
            st.plotly_chart(fig)

            st.session_state.df = df

    if text_input:
        label, score = analyze_text(text_input)
        st.success(f"Text Sentiment: {label}")

    if audio_file:
        t, s, sc = analyze_voice(audio_file)
        st.write("Transcription:", t)
        st.write("Voice Sentiment:", s)

        st.session_state.voice_sent = s

    if video_file:
        emo = analyze_video(video_file)
        st.success(f"Video Emotion: {emo}")

        st.session_state.video_emo = emo


# ================= GEMINI =================
with tab2:

    question = st.text_area(
        "Ask Business Question",
        "What improvements should company make?"
    )

    if st.button("Generate Insights"):
        response = gemini_generate(question)
        st.write(response)

    if st.button("Generate Executive Report"):

        context = ""

        if "df" in st.session_state:
            context += st.session_state.df.head().to_string()

        report = gemini_generate(
            f"Create professional business report:\n{context}"
        )

        st.write(report)


# ================= FUSION =================
with tab3:

    text_sent = None
    if text_input:
        text_sent, _ = analyze_text(text_input)

    voice_sent = st.session_state.get("voice_sent")
    video_emo = st.session_state.get("video_emo")

    if st.button("Calculate Fusion Score"):

        result = fusion_score(text_sent, voice_sent, video_emo)

        st.success(f"Overall Customer Emotion: {result}")
