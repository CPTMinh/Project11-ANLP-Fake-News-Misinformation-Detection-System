import streamlit as st
from roberta_inference import RobertaInferencePipeline

# Page config
st.set_page_config(page_title="Fake News Detector", page_icon="📰", layout="centered")

st.title("📰 Fake News & Misinformation Detector")
st.markdown("Enter a news statement below to verify its authenticity using our fine-tuned RoBERTa model.")

# Load model (cached so it doesn't reload on every UI interaction)
@st.cache_resource
def load_pipeline():
    return RobertaInferencePipeline(checkpoint_dir="models/roberta_best", device="cpu")

try:
    with st.spinner("Loading Model..."):
        pipeline = load_pipeline()
    
    # User interaction
    user_input = st.text_area("News Statement:", height=150, placeholder="Type or paste a news snippet here...")
    
    if st.button("Predict 🔍", type="primary"):
        if user_input.strip() == "":
            st.warning("Please enter some text to analyze.")
        else:
            with st.spinner("Analyzing..."):
                result = pipeline.predict(user_input)
                
                # UI Output Formatting
                label = result["label"]
                conf = result["confidence"]
                
                if label == "REAL":
                    st.success(f"**Prediction: {label}** (Confidence: {conf:.2%})")
                else:
                    st.error(f"**Prediction: {label}** (Confidence: {conf:.2%})")
                
                st.write("**Probabilities:**")
                st.json(result["probabilities"])
                
except Exception as e:
    st.error(f"Error loading model: Make sure you have trained the model and it exists in `models/roberta_best`. Details: {e}")