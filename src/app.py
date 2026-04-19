import streamlit as st
from roberta_inference import RobertaInferencePipeline
import os

# Put your API key here for the app, or rely on the environment variable
os.environ["GEMINI_API_KEY"] = "YOUR_GEMINI_API_KEY"

try:
    from agent import FactCheckingAgent
    agent_available = True
except ImportError:
    agent_available = False

# Page config
st.set_page_config(page_title="Fake News Detector", page_icon="📰", layout="centered")

st.title("📰 Fake News & Misinformation Detector")
st.markdown("Enter a news statement below to verify its authenticity using our fine-tuned RoBERTa model.")

# Get the absolute path to the project root directory relative to app.py
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_DIR = os.path.join(BASE_DIR, "models", "roberta_best")

@st.cache_resource
def load_pipeline():
    return RobertaInferencePipeline(checkpoint_dir=MODEL_DIR, device="cpu")

@st.cache_resource
def load_agent():
    return FactCheckingAgent(roberta_model_dir=MODEL_DIR)

try:
    with st.spinner("Loading Model..."):
        pipeline = load_pipeline()
        if agent_available:
            agent = load_agent()
    
    user_input = st.text_area("News Statement:", height=150, placeholder="Type or paste a news snippet here...")
    
    use_agent = st.checkbox("🔍 Enable Deep Fact-Checking Assistant (RAG)")
    
    if st.button("Predict 🔍", type="primary"):
        if user_input.strip() == "":
            st.warning("Please enter some text to analyze.")
        else:
            if use_agent and agent_available:
                with st.spinner("Agent is retrieving facts and reasoning..."):
                    result = agent.verify_statement(user_input)
                    st.markdown("### Agent Conclusion")
                    st.info(result['agent_decision'])
                    
                    with st.expander("Show intermediate steps"):
                        st.markdown("**1. RoBERTa Linguistic Model:**")
                        st.write(result['roberta'])
                        st.markdown("**2. Search Tool Evidence:**")
                        st.write(result['search_evidence'])
            else:
                with st.spinner("Analyzing style..."):
                    result = pipeline.predict(user_input)
                    label = result["label"]
                    conf = result["confidence"]
                    
                    if label == "REAL":
                        st.success(f"**Prediction: {label}** (Confidence: {conf:.2%})")
                    else:
                        st.error(f"**Prediction: {label}** (Confidence: {conf:.2%})")
                    
                    st.write("**Probabilities:**")
                    st.json(result["probabilities"])
                    
except Exception as e:
    st.error(f"Error: {e}")