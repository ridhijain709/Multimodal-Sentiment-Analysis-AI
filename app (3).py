
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
import google.generativeai as genai
import datetime
import metallama # New: Added for metallama integration

# --- Professional UI Setup ---
st.set_page_config(page_title="AI Customer Vision Pro", layout="wide")
st.title('🌟 AI-Powered Customer Sentiment Assistant')
st.markdown('### Solving Enterprise Problems with Multimodal AI')

# --- Sidebar for Setup ---
st.sidebar.image("https://upload.wikimedia.org/wikipedia/commons/5/51/IBM_logo.svg", width=100)
st.sidebar.header("Control Panel")

# --- Initialize session state for df_sample and analysis results ---
if 'df_sample' not in st.session_state:
    st.session_state.df_sample = pd.DataFrame()
if 'text_analysis_results' not in st.session_state:
    st.session_state.text_analysis_results = {'text': '', 'sentiment': 'neutral', 'score': 0.0}
if 'voice_analysis_results' not in st.session_state:
    st.session_state.voice_analysis_results = {'transcribed_text': '', 'sentiment': 'neutral', 'confidence': 0.0, 'pitch_hz': 0.0, 'energy': 0.0}
if 'video_analysis_results' not in st.session_state:
    st.session_state.video_analysis_results = {'dominant_emotion': 'No emotions detected', 'emotion_counts': {}}
if 'history' not in st.session_state:
    st.session_state.history = []

# --- Helper function to add logs to history ---
def add_to_history(event_type, details):
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    st.session_state.history.append({
        "timestamp": timestamp,
        "event_type": event_type,
        "details": details
    })

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

# --- Gemini Integration ---
if "GEMINI_API_KEY" in st.secrets:
    genai.configure(api_key=st.secrets["GEMINI_API_KEY"])
else:
    st.error("GEMINI_API_KEY not found in Streamlit secrets. Please add it.")

@st.cache_resource
def get_gemini_model():
    return genai.GenerativeModel('gemini-pro')

gemini_model = get_gemini_model()

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

def generate_insights_gpt2(prompt, max_length=150):
    inputs = generator_tokenizer.encode(prompt, return_tensors="pt")
    outputs = generator_model.generate(inputs, max_length=max_length, num_return_sequences=1, no_repeat_ngram_size=2, top_p=0.95)
    return generator_tokenizer.decode(outputs[0], skip_special_tokens=True)

# --- New Gemini Powered Functions ---
def chatbot_response(user_query):
    prompt = f"You are a helpful AI assistant specialized in customer sentiment. Based on the customer data provided and your knowledge, answer the following question: {user_query}"
    try:
        response = gemini_model.generate_content(prompt)
        return response.text
    except Exception as e:
        return f"Error generating chatbot response: {e}"

def calculate_fusion_score(text_sentiment, text_score, voice_sentiment, voice_score, video_dominant_emotion, video_emotion_counts):
    sentiment_map = {'positive': 1, 'neutral': 0, 'negative': -1}

    text_val = sentiment_map.get(text_sentiment.lower(), 0) * text_score
    voice_val = sentiment_map.get(voice_sentiment.lower(), 0) * voice_score

    video_val = 0
    if video_dominant_emotion != 'No emotions detected' and video_emotion_counts:
        emotion_to_sentiment = {
            'happy': 1,
            'neutral': 0,
            'surprise': 0.5,
            'sad': -1,
            'angry': -1,
            'fear': -0.5,
            'disgust': -1
        }
        total_frames = sum(video_emotion_counts.values())
        if total_frames > 0:
            for emotion, count in video_emotion_counts.items():
                video_val += (emotion_to_sentiment.get(emotion, 0) * count) / total_frames

    scores = [s for s in [text_val, voice_val, video_val] if s is not None]
    if scores:
        fusion_score = np.mean(scores)
        overall_sentiment = "Positive" if fusion_score > 0.2 else ("Negative" if fusion_score < -0.2 else "Neutral")
        return f"Fusion Score: {fusion_score:.2f} ({overall_sentiment})"
    return "Not enough data for fusion score."

def generate_report(df_sample_data, text_results, voice_results, video_results):
    report_prompt = "Generate a comprehensive business report based on the following customer sentiment data:\n\n"

    if not df_sample_data.empty:
        report_prompt += "--- CSV Data Summary ---\n"
        sentiment_dist = df_sample_data['predicted_sentiment'].value_counts(normalize=True).to_dict()
        report_prompt += f"Overall CSV Sentiment Distribution: {sentiment_dist}\n"
        negative_reviews = df_sample_data[df_sample_data['predicted_sentiment'] == 'NEGATIVE']['review_text'].head(5).tolist()
        if negative_reviews:
            report_prompt += f"Sample Negative Reviews: {'; '.join(negative_reviews)}\n"
        else:
            report_prompt += "No negative reviews in sample.\n"

    if text_results and text_results['text']:
        report_prompt += "--- Live Text Analysis ---\n"
        report_prompt += f"Text: '{text_results['text']}' -> Sentiment: {text_results['sentiment']} (Score: {text_results['score']:.2f})\n"

    if voice_results and voice_results['transcribed_text'] and voice_results['transcribed_text'] != 'Could not understand audio':
        report_prompt += "--- Live Voice Analysis ---\n"
        report_prompt += f"Transcribed: '{voice_results['transcribed_text']}' -> Sentiment: {voice_results['sentiment']} (Score: {voice_results['confidence']:.2f}), Pitch: {voice_results['pitch_hz']:.2f}, Energy: {voice_results['energy']:.2f}\n"

    if video_results and video_results['dominant_emotion'] != 'No emotions detected':
        report_prompt += "--- Live Video Analysis ---\n"
        report_prompt += f"Dominant Emotion: {video_results['dominant_emotion']}, Emotion Counts: {video_results['emotion_counts']}\n"

    report_prompt += "\nProvide actionable recommendations and a strategic overview.\n"

    try:
        response = gemini_model.generate_content(report_prompt)
        return response.text
    except Exception as e:
        return f"Error generating report: {e}"


# --- New Metallama Functions ---
def get_metallama_insights(text_input):
    try:
        # Example: Using metallama for enhanced text analysis
        # This is a placeholder for actual metallama model calls
        analysis_result = metallama.analyze_text(text_input)
        return f"Metallama Analysis: {analysis_result}"
    except Exception as e:
        return f"Error with Metallama: {e}"


# --- Streamlit UI Input Widgets ---

# Main Tab Input: CSV Upload (in sidebar to be globally accessible for tabs)
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
tab1, tab2, tab3, tab4, tab5, tab6, tab7 = st.tabs(["📊 CSV Dashboard", "📝 Live Text Analysis", "🎙️ Live Voice Analysis", "🎥 Live Video Analysis", "💡 Business Intelligence", "🧠 Advanced AI Tools", "🔮 Metallama Features"])

# --- Tab 1: CSV Dashboard (Comprehensive Analysis) ---
with tab1:
    st.header('Comprehensive Customer Sentiment Dashboard')
    if uploaded_file is not None:
        try:
            df_uploaded = pd.read_csv(uploaded_file)

            # Create a sample of the DataFrame for faster processing
            sample_size = min(5000, len(df_uploaded))
            df_sample = df_uploaded.sample(n=sample_size, random_state=42).copy()
            st.session_state.df_sample = df_sample # Save to session state

            st.write(f"Analyzing a sample of {sample_size} reviews from the uploaded CSV...")

            # Apply sentiment analysis
            review_texts_sample = df_sample['review_text'].tolist()
            batch_results_sample = text_sentiment_pipeline(review_texts_sample)
            predicted_labels_sample = [result['label'] for result in batch_results_sample]
            predicted_scores_sample = [result['score'] for result in batch_results_sample]

            df_sample['predicted_sentiment'] = predicted_labels_sample
            df_sample['sentiment_score'] = predicted_scores_sample
            df_sample['sentiment_match'] = df_sample['sentiment'] == df_sample['predicted_sentiment'].str.lower()

            st.subheader('Sampled Data with Sentiment Analysis Results:')
            st.dataframe(df_sample[['review_text', 'sentiment', 'predicted_sentiment', 'sentiment_score', 'sentiment_match']].head(10))

            add_to_history(
                "CSV Analysis",
                {
                    "filename": uploaded_file.name,
                    "shape": str(df_uploaded.shape),
                    "sample_size": sample_size,
                    "overall_sentiment_distribution": df_sample['predicted_sentiment'].value_counts(normalize=True).to_dict()
                }
            )

            st.subheader('Visualizations from CSV Data:')

            # Graph 1: Sentiment Distribution Pie Chart
            fig1 = px.pie(df_sample, names='predicted_sentiment', title='Overall Sentiment Distribution', hole=0.3)
            st.plotly_chart(fig1, use_container_width=True)

            # Graph 2: Sentiment by Product Category (Bar Chart)
            plt.figure(figsize=(12, 6))
            sns.countplot(data=df_sample, x='product_category', hue='predicted_sentiment', palette='viridis')
            plt.title('Sentiment by Product Category')
            plt.xticks(rotation=45)
            st.pyplot(plt)

            # Graph 3: Average Rating by Region (Box Plot)
            plt.figure(figsize=(10, 6))
            sns.boxplot(data=df_sample, x='region', y='customer_rating', hue='region', palette='Set2', legend=False)
            plt.title('Average Customer Rating by Region')
            st.pyplot(plt)

            # Graph 4: Response Time vs. Issue Resolved (Scatter Plot)
            fig4 = px.scatter(df_sample, x='response_time_hours', y='customer_rating', color='issue_resolved',
                              title='Response Time vs. Rating (Colored by Issue Resolved)',
                              hover_data=['product_category', 'sentiment'])
            st.plotly_chart(fig4, use_container_width=True)

            # Graph 5: Complaint Registered by Age Group (Stacked Bar)
            age_complaints = df_sample.groupby(['age_group', 'complaint_registered']).size().unstack()
            plt.figure(figsize=(10, 6))
            age_complaints.plot(kind='bar', stacked=True)
            plt.title('Complaints by Age Group')
            plt.ylabel('Count')
            st.pyplot(plt)

        except Exception as e:
            st.error(f"Error processing CSV file: {e}")
    else:
        st.info("Please upload a CSV file in the sidebar to see the comprehensive dashboard.")

# --- Tab 2: Live Text Analysis ---
with tab2:
    st.header('Live Text Sentiment Analysis')
    if text_input:
        text_sentiment_label, text_sentiment_score = analyze_text_sentiment(text_input)
        st.session_state.text_analysis_results = {'text': text_input, 'sentiment': text_sentiment_label, 'score': text_sentiment_score}
        st.write(f"**Input Text:** {text_input}")
        st.write(f"**Predicted Sentiment:** {text_sentiment_label}")
        st.write(f"**Confidence Score:** {text_sentiment_score:.4f}")
        add_to_history(
            "Text Analysis",
            {
                "input": text_input,
                "sentiment": text_sentiment_label,
                "score": f"{text_sentiment_score:.4f}"
            }
        )
    else:
        st.info("Enter some text in the 'Live Text Sentiment' box above to get instant analysis.")

# --- Tab 3: Live Voice Analysis ---
with tab3:
    st.header('Live Voice Sentiment Analysis')
    if voice_file is not None:
        try:
            audio_bytes = voice_file.read()
            voice_results = analyze_voice_sentiment(io.BytesIO(audio_bytes))
            st.session_state.voice_analysis_results = voice_results
            st.write(f"**Transcribed Text:** {voice_results['transcribed_text']}")
            st.write(f"**Predicted Sentiment:** {voice_results['sentiment']}")
            st.write(f"**Confidence Score:** {voice_results['confidence']:.4f}")
            st.write(f"**Average Pitch (Hz):** {voice_results['pitch_hz']:.2f}")
            st.write(f"**Average Energy:** {voice_results['energy']:.4f}")
            add_to_history(
                "Voice Analysis",
                {
                    "filename": voice_file.name,
                    "transcribed_text": voice_results['transcribed_text'],
                    "sentiment": voice_results['sentiment'],
                    "confidence": f"{voice_results['confidence']:.4f}"
                }
            )
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
            st.session_state.video_analysis_results = {'dominant_emotion': dominant, 'emotion_counts': counts}

            st.write(f"**Dominant Emotion:** {dominant}")
            st.write("**Emotion Counts:**", counts)

            add_to_history(
                "Video Analysis",
                {
                    "filename": video_file.name,
                    "dominant_emotion": dominant,
                    "emotion_counts": counts
                }
            )

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

# --- Tab 5: Business Intelligence (Gemini) ---
with tab5:
    st.header('💡 Gemini-Powered Business Intelligence')

    if 'df_sample' in st.session_state and not st.session_state.df_sample.empty:
        df_sample_for_gemini = st.session_state.df_sample
        st.subheader("Insights from Uploaded CSV Data")

        # 1. Generate Business Insights (general prompt)
        st.markdown("#### General Business Insights")
        business_prompt_input = st.text_area("Enter a general prompt for business insights (e.g., 'What are key trends in customer feedback?')", "What are the key opportunities for improving customer satisfaction based on recent reviews?")
        if st.button("Generate Business Insights"):
            with st.spinner("Generating business insights with Gemini..."):
                insights = generate_business_insights(business_prompt_input)
                st.write(insights)
                add_to_history(
                    "Gemini Business Insights",
                    {
                        "prompt": business_prompt_input,
                        "response_summary": insights[:100] + "..." if len(insights) > 100 else insights
                    }
                )

        # 2. Generate Customer Personas
        st.markdown("#### Generate Customer Personas")
        persona_reviews_option = st.radio("Select reviews for persona generation:", ("Random Sample from all reviews", "Negative Reviews Sample", "Positive Reviews Sample"), key='persona_option')
        persona_reviews_list = []
        if persona_reviews_option == "Random Sample from all reviews":
            if not df_sample_for_gemini.empty:
                persona_reviews_list = df_sample_for_gemini['review_text'].sample(min(5, len(df_sample_for_gemini)), random_state=42).tolist()
        elif persona_reviews_option == "Negative Reviews Sample":
            negative_reviews_df = df_sample_for_gemini[df_sample_for_gemini['predicted_sentiment'] == 'NEGATIVE']
            if not negative_reviews_df.empty:
                persona_reviews_list = negative_reviews_df['review_text'].sample(min(5, len(negative_reviews_df)), random_state=42).tolist()
        elif persona_reviews_option == "Positive Reviews Sample":
            positive_reviews_df = df_sample_for_gemini[df_sample_for_gemini['predicted_sentiment'] == 'POSITIVE']
            if not positive_reviews_df.empty:
                persona_reviews_list = positive_reviews_df['review_text'].sample(min(5, len(positive_reviews_df)), random_state=42).tolist()

        if persona_reviews_list and st.button("Generate Persona", key='generate_persona_button'):
            with st.spinner("Generating customer persona with Gemini..."):
                persona = generate_personas(persona_reviews_list)
                st.write("**Reviews used for Persona:**")
                for r in persona_reviews_list:
                    st.markdown(f"- {r}")
                st.write("\n**Generated Persona:**")
                st.write(persona)
                add_to_history(
                    "Gemini Persona Generation",
                    {
                        "reviews_used": persona_reviews_list,
                        "persona_summary": persona[:100] + "..." if len(persona) > 100 else persona
                    }
                )
        elif not persona_reviews_list and st.button("Generate Persona (No Reviews Available)", key='no_persona_button', disabled=True):
            st.info("No reviews available for selected option to generate persona.")


        # 3. Extract Complaint Topics from Negative Reviews
        st.markdown("#### Extract Complaint Topics")
        negative_reviews_for_topics = df_sample_for_gemini[df_sample_for_gemini['predicted_sentiment'] == 'NEGATIVE']['review_text'].tolist()
        if negative_reviews_for_topics and st.button("Extract Complaint Topics", key='extract_topics_button'):
            with st.spinner("Extracting complaint topics with Gemini..."):
                topics = extract_complaint_topics(negative_reviews_for_topics[:10]) # Use a subset for prompt length
                st.write("**Sample Negative Reviews analyzed:**")
                for r in negative_reviews_for_topics[:5]:
                    st.markdown(f"- {r}")
                st.write("\n**Extracted Complaint Topics:**")
                st.write(topics)
                add_to_history(
                    "Gemini Complaint Topics",
                    {
                        "sample_negative_reviews": negative_reviews_for_topics[:5],
                        "topics_summary": topics[:100] + "..." if len(topics) > 100 else topics
                    }
                )
        elif not negative_reviews_for_topics and st.button("Extract Complaint Topics (No Negative Reviews)", key='no_topics_button', disabled=True):
            st.info("No negative reviews found in the sample to extract complaint topics.")

    else:
        st.info("Please upload a CSV file in the sidebar and process it in 'CSV Dashboard' to enable Gemini Business Intelligence.")

# --- Tab 6: Advanced AI Tools ---
with tab6:
    st.header('🧠 Advanced AI Tools')

    st.subheader('AI Chatbot for Queries')
    user_query = st.text_area("Ask the AI about the customer data or sentiment analysis:", "What are the main sentiments observed?")
    if st.button("Get Chatbot Response"):
        if user_query:
            with st.spinner("Generating chatbot response..."):
                context = ""
                if not st.session_state.df_sample.empty:
                    context += f"\nOverall CSV Sentiment Distribution: {st.session_state.df_sample['predicted_sentiment'].value_counts(normalize=True).to_dict()}"
                if st.session_state.text_analysis_results['text']:
                    context += f"\nLast Text Analyzed: '{st.session_state.text_analysis_results['text']}' -> Sentiment: {st.session_state.text_analysis_results['sentiment']}"
                if st.session_state.voice_analysis_results['transcribed_text']:
                    context += f"\nLast Voice Analyzed: '{st.session_state.voice_analysis_results['transcribed_text']}' -> Sentiment: {st.session_state.voice_analysis_results['sentiment']}"
                if st.session_state.video_analysis_results['dominant_emotion'] != 'No emotions detected':
                    context += f"\nLast Video Dominant Emotion: {st.session_state.video_analysis_results['dominant_emotion']}"

                full_query = f"Given this context: {context}. User asks: {user_query}"
                response = chatbot_response(full_query)
                st.write("**AI Chatbot:**")
                st.write(response)
                add_to_history(
                    "AI Chatbot",
                    {
                        "query": user_query,
                        "response_summary": response[:100] + "..." if len(response) > 100 else response
                    }
                )
        else:
            st.info("Please enter a query for the chatbot.")

    st.subheader('Multimodal Emotion Fusion Score')
    if st.button("Calculate Fusion Score"):
        text_s = st.session_state.text_analysis_results['sentiment']
        text_sc = st.session_state.text_analysis_results['score']
        voice_s = st.session_state.voice_analysis_results['sentiment']
        voice_sc = st.session_state.voice_analysis_results['confidence']
        video_de = st.session_state.video_analysis_results['dominant_emotion']
        video_ec = st.session_state.video_analysis_results['emotion_counts']

        fusion_output = calculate_fusion_score(text_s, text_sc, voice_s, voice_sc, video_de, video_ec)
        st.write(f"**{fusion_output}**")
        add_to_history(
            "Fusion Score Calculation",
            {
                "text_sentiment": text_s,
                "voice_sentiment": voice_s,
                "video_dominant_emotion": video_de,
                "fusion_result": fusion_output
            }
        )

    st.subheader('AI-Powered Comprehensive Report')
    if st.button("Generate Comprehensive Report"):
        with st.spinner("Generating comprehensive report with Gemini..."):
            report = generate_report(st.session_state.df_sample,
                                     st.session_state.text_analysis_results,
                                     st.session_state.voice_analysis_results,
                                     st.session_state.video_analysis_results)
            st.write("**Comprehensive Report:**")
            st.write(report)
            add_to_history(
                "Comprehensive Report Generation",
                {
                    "report_summary": report[:200] + "..." if len(report) > 200 else report
                }
            )

    st.subheader('Session History')
    if st.session_state.history:
        st.json(st.session_state.history)
    else:
        st.info("No interactions logged yet.")

# --- Tab 7: Metallama Features ---
with tab7:
    st.header('🔮 Metallama-Powered Features')
    st.info("This section will integrate and demonstrate the capabilities of the 'metallama' library.")

    metallama_text_input = st.text_area("Enter text for Metallama analysis:", "How can we improve our customer service?", key="metallama_text_input")

    if st.button("Get Metallama Insights"):
        if metallama_text_input:
            with st.spinner("Generating Metallama insights..."):
                # Placeholder for actual metallama processing
                insights = get_metallama_insights(metallama_text_input)
                st.write("**Metallama Insights:**")
                st.write(insights)
                add_to_history(
                    "Metallama Analysis",
                    {
                        "input_text": metallama_text_input,
                        "insights_summary": insights[:100] + "..." if len(insights) > 100 else insights
                    }
                )
        else:
            st.warning("Please enter text for Metallama analysis.")
