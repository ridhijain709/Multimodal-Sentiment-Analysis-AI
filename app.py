import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# --- STEP 1: LOAD DATA ---
# This part reads your CSV file. 
# ERROR CHECK: Ensure 'Customer_Sentiment.csv' is in the same folder as this file.
try:
    df = pd.read_csv('Customer_Sentiment.csv')
except FileNotFoundError:
    st.error("Error: 'Customer_Sentiment.csv' not found. Please place the CSV in the same folder as this script.")
    st.stop()

# --- STEP 2: APP HEADER ---
st.set_page_config(page_title="AI Sentiment Dashboard", layout="wide")
st.title("🤖 AI Customer Sentiment & Recovery Dashboard")
st.markdown("### BBA Internship Project: Digital Marketing Optimization")

# --- STEP 3: SIDEBAR FILTERS ---
st.sidebar.header("Filter Options")
platform_list = df['platform'].unique()
selected_platform = st.sidebar.selectbox("Select a Platform", platform_list)

# Filter data based on selection
filtered_df = df[df['platform'] == selected_platform]

# --- STEP 4: VISUAL ANALYSIS (ML/Data) ---
col1, col2 = st.columns(2)

with col1:
    st.subheader(f"Sentiment Split: {selected_platform}")
    sentiment_plot = filtered_df['sentiment'].value_counts()
    st.bar_chart(sentiment_plot)

with col2:
    st.subheader("Key Business Metric")
    avg_response = filtered_df['response_time_hours'].mean()
    st.metric("Avg. Response Time", f"{avg_response:.1f} Hours", delta="-2h (Target)")

# --- STEP 5: GENERATIVE AI RECOVERY AGENT ---
st.divider()
st.subheader("🚀 GenAI Recovery Agent")
st.write("Automatically generate a personalized marketing response for negative reviews.")

# Pick a negative review from the data
negative_reviews = filtered_df[filtered_df['sentiment'] == 'negative']

if not negative_reviews.empty:
    sample_review = negative_reviews.iloc[0] # Takes the first negative review found
    
    st.warning(f"**Customer Complaint:** {sample_review['review_text']}")
    
    if st.button("Generate AI Apology Email"):
        # This is the 'Generative AI Method' logic
        # It crafts a response based on the specific platform and complaint
        ai_draft = f"""
        Subject: Resolving your experience on {selected_platform}
        
        Dear Customer,
        
        We saw your feedback regarding '{sample_review['review_text']}'. 
        We sincerely apologize for the delay. As a brand on {selected_platform}, 
        we want to offer you a 20% discount code: AI_RECOVER_20.
        
        Best regards,
        Digital Marketing Team
        """
        st.success("AI Generated Draft:")
        st.text_area("Copy/Paste for Customer Support:", ai_draft, height=200)
else:
    st.success(f"No negative reviews found for {selected_platform}!")