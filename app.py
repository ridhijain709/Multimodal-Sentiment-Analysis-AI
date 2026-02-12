
import streamlit as st
import pandas as pd
import plotly.express as px
import matplotlib.pyplot as plt
import seaborn as sns
import torch
from transformers import pipeline, GPT2LMHeadModel, GPT2Tokenizer
import speech_recognition as sr
import librosa
import cv2
from fer.fer import FER
import numpy as np
import io
import tempfile
import os
from gtts import gTTS
from pydub import AudioSegment

# --- Professional UI Setup ---
st.set_page_config(page_title="AI Customer Vision Pro", layout="wide")
st.title('🌟 AI-Powered Customer Sentiment Assistant')
st.markdown('### Solving Enterprise Problems with Multimodal AI')

# --- Sidebar for Setup ---
st.sidebar.image("https://upload.wikimedia.org/wikipedia/commons/5/51/IBM_logo.svg", width=100)
st.sidebar.header("Control Panel")

# --- Caching expensive models ---
@st.cache_resource
def get_sentiment_pipeline():
    return pipeline("sentiment-analysis", model="distilbert-base-uncased-finetuned-sst-2-english")

@st.cache_resource
def get_gpt2_models():
    tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
    model = GPT2LMHeadModel.from_pretrained("gpt2")
    return tokenizer, model

text_sentiment_pipeline = get_sentiment_pipeline()
generator_tokenizer, generator_model = get_gpt2_models()

# --- Define AI Functions (re-integrated from comprehensive notebook) ---
def analyze_text_sentiment(text):
    if pd.isna(text):
        return "No review", 0.0
    result = text_sentiment_pipeline(text)[0]
    return result['label'], result['score']

def analyze_voice_sentiment(audio_input):
    temp_audio_path = None
    if isinstance(audio_input, io.BytesIO):
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp_file:
            tmp_file.write(audio_input.read())
            temp_audio_path = tmp_file.name
        audio_file_to_process = temp_audio_path
    else:
        audio_file_to_process = audio_input

    transcribed_text = "Could not understand audio"
    sentiment_label = "neutral"
    sentiment_score = 0.0
    pitch = 0.0
    energy = 0.0

    try:
        y, sr_rate = librosa.load(audio_file_to_process, sr=None)
        recognizer = sr.Recognizer()
        with sr.AudioFile(audio_file_to_process) as source:
            audio_data = recognizer.record(source)
            try:
                transcribed_text = recognizer.recognize_google(audio_data)
            except sr.UnknownValueError:
                transcribed_text = "Could not understand audio"

        sentiment_label, sentiment_score = analyze_text_sentiment(transcribed_text)

        try:
            pitches, magnitudes = librosa.core.piptrack(y=y, sr=sr_rate)
            pitch = np.mean(pitches[pitches > 0]) if np.any(pitches > 0) else 0.0
        except Exception:
            pitch = 0.0

        energy = np.mean(librosa.feature.rms(y=y)) if len(y) > 0 else 0.0

    except Exception as e:
        st.warning(f"Error processing audio: {e}")
    finally:
        if temp_audio_path and os.path.exists(temp_audio_path):
            os.remove(temp_audio_path)

    return {
        'transcribed_text': transcribed_text,
        'sentiment': sentiment_label,
        'confidence': sentiment_score,
        'pitch_hz': pitch,
        'energy': energy
    }

def analyze_video_sentiment(video_file):
    detector = FER(mtcnn=True)
    cap = cv2.VideoCapture(video_file)
    emotions = []
    frame_count = 0
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        frame_count += 1
        if frame_count % 30 == 0:
            result = detector.detect_emotions(frame)
            if result:
                if result[0]['emotions']:
                    top_emotion = max(result[0]['emotions'], key=result[0]['emotions'].get)
                    emotions.append(top_emotion)
    cap.release()
    if emotions:
        dominant_emotion = max(set(emotions), key=emotions.count)
        emotion_counts = {emo: emotions.count(emo) for emo in set(emotions)}
        return dominant_emotion, emotion_counts
    return "No emotions detected", {}

def generate_insights(prompt, max_length=150):
    inputs = generator_tokenizer.encode(prompt, return_tensors="pt")
    outputs = generator_model.generate(inputs, max_length=max_length, num_return_sequences=1, no_repeat_ngram_size=2, top_p=0.95)
    return generator_tokenizer.decode(outputs[0], skip_special_tokens=True)

# --- Streamlit UI Input Widgets ---

# Main Tab Input: CSV Upload
uploaded_file = st.sidebar.file_uploader("Upload your Customer_Sentiment.csv file", type=["csv"])

# Live Input Widgets (in main area for better visibility)
st.header('Analyze New Data')
col_text, col_voice, col_video = st.columns(3)

with col_text:
    st.subheader('Live Text Sentiment')
    text_input = st.text_area("Enter text for sentiment analysis:", "This product is fantastic!")

with col_voice:
    st.subheader('Live Voice Sentiment')
    voice_file = st.file_uploader("Upload an audio file (WAV) for analysis:", type=["wav"])

with col_video:
    st.subheader('Live Video Sentiment')
    video_file = st.file_uploader("Upload a video file (MP4) for analysis:", type=["mp4"])


# --- Create Tabs for comprehensive interface ---
tab1, tab2, tab3, tab4 = st.tabs(["📊 CSV Dashboard", "📝 Live Text Analysis", "🎙️ Live Voice Analysis", "🎥 Live Video Analysis"])

# --- Tab 1: CSV Dashboard (Comprehensive Analysis) ---
with tab1:
    st.header('Comprehensive Customer Sentiment Dashboard')
    csv_file = st.file_uploader("Upload CSV", type=["csv"])

    if csv_file:
        try:
            df = pd.read_csv(csv_file)
            selected_column = st.selectbox("Select text column", df.columns)

            df[selected_column] = df[selected_column].astype(str)

            # Create a sample of the DataFrame for faster processing
            sample_size = min(5000, len(df))
            df_sample = df.sample(n=sample_size, random_state=42).copy()

            st.write(f"Analyzing a sample of {sample_size} reviews from the uploaded CSV.")

            # Apply sentiment analysis
            review_texts_sample = df_sample[selected_column].tolist()
            batch_results_sample = text_sentiment_pipeline(review_texts_sample)
            predicted_labels_sample = [result['label'] for result in batch_results_sample]
            predicted_scores_sample = [result['score'] for result in batch_results_sample]

            df_sample['predicted_sentiment'] = predicted_labels_sample
            df_sample['sentiment_score'] = predicted_scores_sample
            # Only try to compare if 'sentiment' column exists in the original df_uploaded
            if 'sentiment' in df_sample.columns:
                df_sample['sentiment_match'] = df_sample['sentiment'] == df_sample['predicted_sentiment'].str.lower()
                st.subheader('Sampled Data with Sentiment Analysis Results:')
                st.dataframe(df_sample[[selected_column, 'sentiment', 'predicted_sentiment', 'sentiment_score', 'sentiment_match']].head(10))
            else:
                st.subheader('Sampled Data with Predicted Sentiment Results:')
                st.dataframe(df_sample[[selected_column, 'predicted_sentiment', 'sentiment_score']].head(10))

            st.subheader('Visualizations from CSV Data:')

            # Graph 1: Sentiment Distribution Pie Chart
            fig1 = px.pie(df_sample, names='predicted_sentiment', title='Overall Sentiment Distribution', hole=0.3)
            st.plotly_chart(fig1, use_container_width=True)

            # Ensure 'product_category', 'region', 'customer_rating', 'response_time_hours', 'issue_resolved', 'age_group', 'complaint_registered' exist before plotting
            if 'product_category' in df_sample.columns and 'predicted_sentiment' in df_sample.columns:
                # Graph 2: Sentiment by Product Category (Bar Chart)
                plt.figure(figsize=(12, 6))
                sns.countplot(data=df_sample, x='product_category', hue='predicted_sentiment', palette='viridis')
                plt.title('Sentiment by Product Category')
                plt.xticks(rotation=45)
                st.pyplot(plt)

            if 'region' in df_sample.columns and 'customer_rating' in df_sample.columns:
                # Graph 3: Average Rating by Region (Box Plot)
                plt.figure(figsize=(10, 6))
                sns.boxplot(data=df_sample, x='region', y='customer_rating', hue='region', palette='Set2', legend=False)
                plt.title('Average Customer Rating by Region')
                st.pyplot(plt)

            if 'response_time_hours' in df_sample.columns and 'customer_rating' in df_sample.columns and 'issue_resolved' in df_sample.columns:
                # Graph 4: Response Time vs. Issue Resolved (Scatter Plot)
                fig4 = px.scatter(df_sample, x='response_time_hours', y='customer_rating', color='issue_resolved',
                                  title='Response Time vs. Rating (Colored by Issue Resolved)',
                                  hover_data=['product_category', 'sentiment'])
                st.plotly_chart(fig4, use_container_width=True)

            if 'age_group' in df_sample.columns and 'complaint_registered' in df_sample.columns:
                # Graph 5: Complaint Registered by Age Group (Stacked Bar)
                age_complaints = df_sample.groupby(['age_group', 'complaint_registered']).size().unstack()
                plt.figure(figsize=(10, 6))
                age_complaints.plot(kind='bar', stacked=True)
                plt.title('Complaints by Age Group')
                plt.ylabel('Count')
                st.pyplot(plt)

            # Generative AI Insights
            st.subheader('Generative AI Insights from CSV Data')
            negative_reviews_df = df_sample[df_sample['predicted_sentiment'] == 'NEGATIVE'][selected_column]
            if not negative_reviews_df.empty:
                sample_size_neg = min(5, len(negative_reviews_df))
                negative_reviews = negative_reviews_df.sample(sample_size_neg, random_state=42).tolist()
                if negative_reviews:
                    prompt = f"Based on these negative customer reviews: {', '.join(negative_reviews)}\nSuggest improvements for the brand:"
                    insights = generate_insights(prompt)
                    st.write("**Insights from Negative Reviews:**")
                    st.write(insights)
                else:
                    st.write("No negative reviews found in the sample to generate insights.")
            else:
                st.write("No negative reviews found in the sample to generate insights.")

            positive_count = len(df_sample[df_sample['predicted_sentiment']=='POSITIVE'])
            negative_count = len(df_sample[df_sample['predicted_sentiment']=='NEGATIVE'])
            neutral_count = len(df_sample[df_sample['predicted_sentiment']=='NEUTRAL'])
            overall_prompt = f"Summary of sentiments: Positive: {positive_count}, Negative: {negative_count}, Neutral: {neutral_count}.\nKey recommendations:"
            summary_insights = generate_insights(overall_prompt)
            st.write("**Overall Summary Insights:**")
            st.write(summary_insights)

        except Exception as e:
            st.error(f"Error processing CSV file: {e}")
    else:
        st.info("Please upload a CSV file in the sidebar to see the comprehensive dashboard.")

# --- Tab 2: Live Text Analysis ---
with tab2:
    st.header('Live Text Sentiment Analysis')
    if text_input:
        text_sentiment_label, text_sentiment_score = analyze_text_sentiment(text_input)
        st.write(f"**Input Text:** {text_input}")
        st.write(f"**Predicted Sentiment:** {text_sentiment_label}")
        st.write(f"**Confidence Score:** {text_sentiment_score:.4f}")
    else:
        st.info("Enter some text in the 'Live Text Sentiment' box above to get instant analysis.")

# --- Tab 3: Live Voice Analysis ---
with tab3:
    st.header('Live Voice Sentiment Analysis')
    if voice_file is not None:
        try:
            audio_bytes = voice_file.read()
            voice_results = analyze_voice_sentiment(io.BytesIO(audio_bytes))
            st.write(f"**Transcribed Text:** {voice_results['transcribed_text']}")
            st.write(f"**Predicted Sentiment:** {voice_results['sentiment']}")
            st.write(f"**Confidence Score:** {voice_results['confidence']:.4f}")
            st.write(f"**Average Pitch (Hz):** {voice_results['pitch_hz']:.2f}")
            st.write(f"**Average Energy:** {voice_results['energy']:.4f}")
        except Exception as e:
            st.error(f"Error processing voice file: {e}")
    else:
        st.info("Upload a WAV audio file in the 'Live Voice Sentiment' box above to analyze speech.")

# --- Tab 4: Live Video Analysis ---
with tab4:
    st.header('Live Video Sentiment Analysis')
    if video_file is not None:
        try:
            with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as tmp_file:
                tmp_file.write(video_file.read())
                temp_video_path = tmp_file.name

            st.write(f"Analyzing video file: {video_file.name}")
            dominant, counts = analyze_video_sentiment(temp_video_path)

            st.write(f"**Dominant Emotion:** {dominant}")
            st.write("**Emotion Counts:**", counts)

            if counts:
                plt.figure(figsize=(8, 4))
                sns.barplot(x=list(counts.keys()), y=list(counts.values()), hue=list(counts.keys()), palette='coolwarm', legend=False)
                plt.title('Emotion Distribution in Video')
                plt.xlabel('Emotions')
                plt.ylabel('Frame Count')
                st.pyplot(plt)
            else:
                st.write("No emotions detected in the video.")

        except Exception as e:
            st.error(f"Error processing video file: {e}")
        finally:
            if 'temp_video_path' in locals() and os.path.exists(temp_video_path):
                os.remove(temp_video_path)
    else:
        st.info("Upload an MP4 video file in the 'Live Video Sentiment' box above to analyze facial emotions.")
