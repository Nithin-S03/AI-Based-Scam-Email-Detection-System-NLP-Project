import streamlit as st
import pandas as pd
import joblib
import re
import nltk
import time
import plotly.graph_objects as go
import os

# Download NLTK resources
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')
    nltk.download('punkt_tab')
try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('stopwords')

from src.train_model import preprocess_text

# Check if running with streamlit
import sys
if "streamlit" not in sys.modules:
    print("\n[!] ERROR: This application must be run using Streamlit.")
    print("[!] Please run: python -m streamlit run app.py\n")
    sys.exit(1)

# Page Configuration
st.set_page_config(
    page_title="AI Email Phishing Detector",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for modern look and animations
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;600;700&display=swap');
    
    html, body, [class*="css"] {
        font-family: 'Inter', sans-serif;
    }
    
    .stApp {
        background: linear-gradient(135deg, #0f0c29 0%, #302b63 50%, #24243e 100%);
        color: white;
    }
    
    .main-header {
        text-align: center;
        padding: 2rem;
        background: rgba(255, 255, 255, 0.05);
        backdrop-filter: blur(10px);
        border-radius: 15px;
        margin-bottom: 2rem;
        border: 1px solid rgba(255, 255, 255, 0.1);
    }
    
    .result-card {
        padding: 1.5rem;
        border-radius: 12px;
        background: rgba(255, 255, 255, 0.08);
        border: 1px solid rgba(255, 255, 255, 0.15);
        margin-top: 1rem;
    }
    
    .phishing-label {
        color: #ff4b4b;
        font-weight: 700;
        font-size: 1.2rem;
    }
    
    .safe-label {
        color: #00ff88;
        font-weight: 700;
        font-size: 1.2rem;
    }
    
    .suspicious-word {
        background-color: rgba(255, 75, 75, 0.3);
        padding: 2px 4px;
        border-radius: 4px;
        border-bottom: 2px solid #ff4b4b;
    }

    /* Floating Chatbot */
    #chatbot-toggle {
        position: fixed;
        bottom: 20px;
        right: 20px;
        z-index: 1000;
        background: #00ff88;
        color: black;
        width: 60px;
        height: 60px;
        border-radius: 50%;
        display: flex;
        align-items: center;
        justify-content: center;
        cursor: pointer;
        box-shadow: 0 4px 15px rgba(0,255,136,0.3);
        transition: transform 0.3s ease;
    }
    
    #chatbot-toggle:hover {
        transform: scale(1.1);
    }
</style>
""", unsafe_allow_stdio=True, unsafe_allow_html=True)

# Load Model and Vectorizer
@st.cache_resource
def load_assets():
    try:
        model = joblib.load("models/phishing_model.pkl")
        tfidf = joblib.load("models/tfidf_vectorizer.pkl")
        return model, tfidf
    except Exception as e:
        st.error(f"Error loading model: {e}")
        return None, None

model, tfidf = load_assets()

# Sidebar Navigation
st.sidebar.title("🛡️ Cyber Guardian")
page = st.sidebar.radio("Navigation", ["Phishing Detection", "Security Assistant Chatbot"])

# Floating Chatbot State
if 'chat_open' not in st.session_state:
    st.session_state.chat_open = False
if 'messages' not in st.session_state:
    st.session_state.messages = [{"role": "assistant", "content": "Hello! I am your Email Security Assistant. Ask me anything about phishing emails."}]

# Helper Functions
def get_risk_level(score):
    if score > 0.8: return "High Risk", "🔴"
    if score > 0.4: return "Medium Risk", "🟡"
    return "Low Risk", "🟢"

def highlight_suspicious(text):
    keywords = ["urgent", "account", "suspended", "verify", "password", "bank", "click", "link", "offer", "prize", "limited", "action required", "official"]
    highlighted = text
    for word in keywords:
        pattern = re.compile(re.escape(word), re.IGNORECASE)
        highlighted = pattern.sub(f'<span class="suspicious-word">{word}</span>', highlighted)
    return highlighted

# --- PAGE 1: PHISHING DETECTION ---
if page == "Phishing Detection":
    st.markdown('<div class="main-header"><h1>AI Email Phishing Detection System</h1><p>Powered by NLP & Machine Learning</p></div>', unsafe_allow_html=True)
    
    email_text = st.text_area("Paste the Email Content Here", placeholder="Enter the full text of the email you received...", height=300)
    
    if st.button("Analyze Email"):
        if not email_text.strip():
            st.warning("Please paste an email message for analysis.")
        else:
            with st.spinner("Processing email analysis..."):
                time.sleep(1.5) # Simulate processing
                
                # Preprocess and predict
                clean_text = preprocess_text(email_text)
                vectorized = tfidf.transform([clean_text])
                prediction = model.predict(vectorized)[0]
                probabilities = model.predict_proba(vectorized)[0]
                confidence = probabilities[prediction]
                
                # Layout for results
                col1, col2 = st.columns([1, 1])
                
                with col1:
                    st.subheader("Analysis Results")
                    if prediction == 1:
                        st.markdown('<p class="phishing-label">🚩 PHISHING EMAIL DETECTED</p>', unsafe_allow_html=True)
                        risk, icon = get_risk_level(confidence)
                    else:
                        st.markdown('<p class="safe-label">✅ SAFE EMAIL</p>', unsafe_allow_html=True)
                        risk, icon = "Low Risk", "🟢"
                    
                    st.write(f"**Confidence Score:** {confidence*100:.2f}%")
                    st.write(f"**Risk Level:** {risk} {icon}")
                    
                    # Gauge Chart
                    fig = go.Figure(go.Indicator(
                        mode = "gauge+number",
                        value = confidence * 100,
                        domain = {'x': [0, 1], 'y': [0, 1]},
                        title = {'text': "Confidence Score"},
                        gauge = {
                            'axis': {'range': [0, 100]},
                            'bar': {'color': "#00ff88" if prediction == 0 else "#ff4b4b"},
                            'steps': [
                                {'range': [0, 50], 'color': "rgba(0,0,0,0)"},
                                {'range': [50, 100], 'color': "rgba(0,0,0,0)"}
                            ]
                        }
                    ))
                    fig.update_layout(paper_bgcolor='rgba(0,0,0,0)', font={'color': "white"}, height=250)
                    st.plotly_chart(fig, use_container_width=True)

                with col2:
                    st.subheader("Explanation & Indicators")
                    st.markdown('<div class="result-card">', unsafe_allow_html=True)
                    if prediction == 1:
                        st.error("Suspicious indicators detected:")
                        st.write("- Urgent or threatening language")
                        st.write("- Request for sensitive verification")
                        st.write("- Potential suspicious links (detected keywords)")
                    else:
                        st.success("Analysis complete:")
                        st.write("- No high-risk keywords detected")
                        st.write("- Language appears consistent with normal communication")
                    st.markdown('</div>', unsafe_allow_html=True)
                    
                    st.markdown("### Text Analysis View")
                    st.markdown(f'<div style="background: rgba(255,255,255,0.05); padding: 10px; border-radius: 5px;">{highlight_suspicious(email_text)}</div>', unsafe_allow_html=True)

# --- PAGE 2: SECURITY ASSISTANT ---
elif page == "Security Assistant Chatbot":
    st.markdown('<div class="main-header"><h1>Email Security Assistant</h1><p>Learn how to stay safe online</p></div>', unsafe_allow_html=True)
    
    st.markdown("""
    ### Common Questions
    - How can I detect phishing emails?
    - What makes an email suspicious?
    - What should I do if I receive a scam email?
    - How do hackers trick people with emails?
    """)
    
    # Simple educational chatbot logic
    def get_assistant_response(user_input):
        responses = {
            "detect": "Check for suspicious sender addresses, urgent language, and links that don't match the official domain.",
            "suspicious": "Look for generic greetings, poor grammar, and requests for private information like passwords.",
            "receive": "Do not click any links. Report the email as spam and delete it immediately.",
            "trick": "Hackers use social engineering to create fear or excitement, making you act quickly without thinking."
        }
        user_input = user_input.lower()
        for key in responses:
            if key in user_input:
                return responses[key]
        return "I'm here to help with email security! You can ask about detecting scams or what to do with suspicious emails."

    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    if prompt := st.chat_input("Ask Cyber Email Guardian..."):
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)
        
        response = get_assistant_response(prompt)
        st.session_state.messages.append({"role": "assistant", "content": response})
        with st.chat_message("assistant"):
            st.markdown(response)

# --- FLOATING CHATBOT UI ---
st.sidebar.markdown("---")
if st.sidebar.button("💬 Open Cyber Guardian"):
    st.session_state.chat_open = not st.session_state.chat_open

if st.session_state.chat_open:
    with st.sidebar.container():
        st.info("Cyber Email Guardian (Floating Preview)")
        st.caption("I'm always here to help on both pages!")
        # Re-using simple logic for sidebar preview
        # This is a simplified version of the floating UI within Streamlit's constraints
