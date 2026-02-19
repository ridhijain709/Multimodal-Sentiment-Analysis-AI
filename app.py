import streamlit as st
import pandas as pd
import plotly.express as px
import matplotlib.pyplot as plt
import seaborn as sns
import torch
from transformers import pipeline
from transformers import GPT2LMHeadModel
from transformers import GPT2Tokenizer
import speech_recognition as sr
import librosa
import cv2
from fer.fer import FER
import numpy as np
import io
import tempfile
import os
import google.generativeai as genai
import datetime

# -------------------------
# PAGE SETUP
# -------------------------
st.set_page_config(page_title="AI Customer Vision Pro", layout="wide")
st.title("🌟 AI-Powered Customer Sentiment Assistant")
st.markdown("### Multimodal AI for Enterprise Insights")

# -------------------------
# SIDEBAR
# -------------------------
st.sidebar.header("Control Panel")

# -------------------------
# SESSION STATE
# -------------------------
if "df_sample" not in st.session_state:
    st.session_state.df_sample = pd.DataFrame()

if "history" not in st.session_state:
    st.session_state.history = []

# -------------------------
# HISTORY LOGGER
# -------------------------
def add_to_history(event_type, details):
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    st.session_state.history.append(
        {"timestamp": timestamp, "event": event_type, "details": details}
    )

# -------------------------
# LOAD MODELS
# -------------------------
@st.cache_resource
def load_sentiment():
    return pipeline(
        "sentiment-analysis",
        model="distilbert-base-uncased-finetuned-sst-2-english"
    )

@st.cache_resource
def load_gpt2():
    tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
    model = GPT2LMHeadModel.from_pretrained("gpt2")
    return tokenizer, model

sentiment_pipeline = load_sentiment()
gpt_tokenizer, gpt_model = load_gpt2()

# -------------------------
# GEMINI SETUP
# -------------------------
if "GEMINI_API_KEY" in st.secrets:
    genai.configure(api_key=st.secrets["GEMINI_API_KEY"])

@st.cache_resource
def load_gemini():
    return genai.GenerativeModel("gemini-1.5-flash")

gemini_model = load_gemini()

# -------------------------
# GEMINI FUNCTIONS
# -------------------------
def generate_business_insights(prompt):
    try:
        response = gemini_model.generate_content(prompt)
        return response.text
    except Exception as e:
        return str(e)


def generate_personas(reviews):
    prompt = f"""
    Create customer personas from these reviews:
    {reviews}
    Include demographics, pain points and expectations.
    """
    return generate_business_insights(prompt)


def extract_complaint_topics(reviews):
    prompt = f"""
    Identify major complaint topics from:
    {reviews}
    Provide categories and causes.
    """
    return generate_business_insights(prompt)


def chatbot_response(question):
    return generate_business_insights(question)

# -------------------------
# ANALYSIS FUNCTIONS
# -------------------------
def analyze_text(text):
    result = sentiment_pipeline(text)[0]
    return result["label"], result["score"]


def analyze_voice(audio_file):
    recognizer = sr.Recognizer()

    with tempfile.NamedTemporaryFile(delete=False) as tmp:
        tmp.write(audio_file.read())
        temp_path = tmp.name

    with sr.AudioFile(temp_path) as source:
        audio_data = recognizer.record(source)
        try:
            text = recognizer.recognize_google(audio_data)
        except:
            text = "Could not understand audio"

    sentiment, score = analyze_text(text)

    y, sr_rate = librosa.load(temp_path)
    energy = float(np.mean(librosa.feature.rms(y=y)))

    os.remove(temp_path)

    return text, sentiment, score, energy


def analyze_video(video_file):
    detector = FER(mtcnn=True)

    with tempfile.NamedTemporaryFile(delete=False) as tmp:
        tmp.write(video_file.read())
        path = tmp.name

    cap = cv2.VideoCapture(path)
    emotions = []

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        result = detector.detect_emotions(frame)
        if result:
            top = max(result[0]["emotions"], key=result[0]["emotions"].get)
            emotions.append(top)

    cap.release()
    os.remove(path)

    if emotions:
        return max(set(emotions), key=emotions.count)
    return "No emotion detected"

# -------------------------
# FUSION SCORE
# -------------------------
def fusion_score(text_sent, voice_sent, video_emotion):

    score_map = {"POSITIVE": 1, "NEGATIVE": -1, "NEUTRAL": 0}

    values = [
        score_map.get(text_sent, 0),
        score_map.get(voice_sent, 0)
    ]

    if video_emotion == "happy":
        values.append(1)
    elif video_emotion in ["sad", "angry"]:
        values.append(-1)

    return np.mean(values)

# -------------------------
# INPUTS
# -------------------------
uploaded_file = st.sidebar.file_uploader("Upload CSV", type=["csv"])

text_input = st.text_area("Enter text")

voice_file = st.file_uploader("Upload voice", type=["wav"])

video_file = st.file_uploader("Upload video", type=["mp4"])

# -------------------------
# TABS
# -------------------------
tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
    "📊 Dashboard",
    "📝 Text",
    "🎙 Voice",
    "🎥 Video",
    "💡 Business AI",
    "🧠 Advanced"
])

# -------------------------
# CSV DASHBOARD
# -------------------------
with tab1:

    if uploaded_file:
        df = pd.read_csv(uploaded_file)

        df_sample = df.sample(min(1000, len(df)))
        sentiments = sentiment_pipeline(df_sample["review_text"].tolist())

        df_sample["predicted"] = [s["label"] for s in sentiments]

        st.session_state.df_sample = df_sample

        fig = px.pie(df_sample, names="predicted")
        st.plotly_chart(fig)

# -------------------------
# TEXT TAB
# -------------------------
with tab2:

    if st.button("Analyze Text") and text_input:

        label, score = analyze_text(text_input)

        st.write(label, score)

# -------------------------
# VOICE TAB
# -------------------------
with tab3:

    if voice_file and st.button("Analyze Voice"):

        text, label, score, energy = analyze_voice(voice_file)

        st.write(text)
        st.write(label, score)
        st.write("Energy:", energy)

# -------------------------
# VIDEO TAB
# -------------------------
with tab4:

    if video_file and st.button("Analyze Video"):

        emotion = analyze_video(video_file)

        st.write("Emotion:", emotion)

# -------------------------
# BUSINESS AI TAB
# -------------------------
with tab5:

    prompt = st.text_area("Ask business question")

    if st.button("Generate Insights"):
        st.write(generate_business_insights(prompt))

# -------------------------
# ADVANCED TAB
# -------------------------
with tab6:

    question = st.text_input("Ask AI")

    if st.button("Chat"):
        st.write(chatbot_response(question))

    st.subheader("Fusion Score")

    if st.button("Calculate Fusion"):

        text_label = "POSITIVE"
        voice_label = "POSITIVE"
        video_emotion = "happy"

        score = fusion_score(text_label, voice_label, video_emotion)

        st.write("Fusion Score:", score)

# -------------------------
# HISTORY
# -------------------------
st.sidebar.subheader("Session History")
st.sidebar.json(st.session_state.history)

