import streamlit as st
import pandas as pd
import plotly.express as px
import google.generativeai as genai
from transformers import pipeline
import speech_recognition as sr
from PIL import Image
import cv2
import tempfile
import os

# --- PAGE CONFIG ---
st.set_page_config(page_title="IBM AI Multimodal Pro v5", layout="wide")
st.title("🚀 AI Customer Intelligence & Strategy Dashboard")

# --- 2026 GEN AI SETUP (404 FIX) ---
if "GEMINI_API_KEY" in st.secrets:
    genai.configure(api_key=st.secrets["GEMINI_API_KEY"])
    # Logic to pick the best available 2026 model
    try:
        available_models = [m.name for m in genai.list_models()]
        if 'models/gemini-3-flash-preview' in available_models:
            model_id = "gemini-3-flash-preview"
        elif 'models/gemini-1.5-flash' in available_models:
            model_id = "gemini-1.5-flash"
        else:
            model_id = "gemini-pro"
        model = genai.GenerativeModel(model_id)
        st.sidebar.success(f"Connected: {model_id}")
    except:
        st.sidebar.warning("API connection issues. Check Secret Key.")
else:
    st.error("Missing API Key in Secrets!")

# --- LOCAL AI ENGINES (NO QUOTA LIMITS) ---
@st.cache_resource
def load_ai_engines():
    # Text Sentiment
    txt_engine = pipeline("sentiment-analysis", model="distilbert-base-uncased-finetuned-sst-2-english")
    # Video Emotion (Replaces broken FER)
    vis_engine = pipeline("image-classification", model="dima806/facial_emotions_image_detection")
    return txt_engine, vis_engine

txt_ai, vis_ai = load_ai_engines()

# --- APP LAYOUT ---
tab1, tab2, tab3 = st.tabs(["🎙️ Multimodal Analysis", "📊 Data Dashboard", "🤖 GenAI Strategy"])

# --- TAB 1: VOICE, VIDEO, TEXT ---
with tab1:
    st.subheader("Live Sentiment Input")
    c1, c2 = st.columns(2)
    
    with c1:
        # Text Input
        u_text = st.text_area("Customer Text Feedback:")
        if u_text:
            res = txt_ai(u_text)[0]
            st.write(f"**Text Mood:** {res['label']} ({res['score']:.2f})")
            st.session_state['t_data'] = f"Text: {res['label']}"

        # Fixed Voice Input (v5)
        v_file = st.file_uploader("Upload Customer Voice (.wav)", type=['wav'])
        if v_file:
            r = sr.Recognizer()
            with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp:
                tmp.write(v_file.getvalue())
                with sr.AudioFile(tmp.name) as source:
                    audio_data = r.record(source)
                    text = r.recognize_google(audio_data)
                    st.success(f"Transcribed: {text}")
                    st.session_state['a_data'] = f"Voice said: {text}"

    with c2:
        # Video Input
        vid_file = st.file_uploader("Upload Customer Video (.mp4)", type=['mp4'])
        if vid_file:
            with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as tmp:
                tmp.write(vid_file.read())
                cap = cv2.VideoCapture(tmp.name)
                ret, frame = cap.read()
                if ret:
                    img = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
                    st.image(img, width=250)
                    emo = vis_ai(img)[0]
                    st.warning(f"Video Emotion: {emo['label']}")
                    st.session_state['v_data'] = f"Video Emotion: {emo['label']}"
                cap.release()

# --- TAB 2: CSV DASHBOARD & GRAPHS ---
with tab2:
    st.subheader("Customer Data Insights")
    csv_file = st.file_uploader("Upload CSV Data", type=['csv'])
    if csv_file:
        df = pd.read_csv(csv_file)
        st.dataframe(df.head(5), use_container_width=True)
        
        # Interactive Plotly Graphs
        col_x = st.selectbox("X-Axis (Category):", df.columns)
        col_y = st.selectbox("Y-Axis (Value):", df.select_dtypes(include='number').columns)
        
        fig = px.bar(df, x=col_x, y=col_y, color=col_x, title=f"{col_y} by {col_x}", template="plotly_dark")
        st.plotly_chart(fig, use_container_width=True)
        st.session_state['csv_data'] = f"CSV Analytics: {len(df)} records analyzed."

# --- TAB 3: GENERATIVE AI ---
with tab3:
    st.subheader("IBM Executive Progress Report")
    if st.button("Generate Final AI Strategy"):
        summary_ctx = f"""
        {st.session_state.get('t_data', '')} 
        {st.session_state.get('a_data', '')} 
        {st.session_state.get('v_data', '')} 
        {st.session_state.get('csv_data', '')}
        """
        prompt = f"As an IBM AI Intern, analyze this multimodal customer data and provide a 3-point strategy: {summary_ctx}"
        
        try:
            with st.spinner("AI is calculating strategy..."):
                response = model.generate_content(prompt)
                st.markdown(response.text)
        except Exception as e:
            st.error(f"Quota reached. Try again in a few minutes or check your API key.")


