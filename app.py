import streamlit as st
import warnings
warnings.filterwarnings("ignore")

# Add error handling for imports
try:
    from simple_attention import SimpleAttentionVisualizer
    MODEL_LOADED = True
except Exception as e:
    MODEL_LOADED = False
    ERROR_MSG = str(e)

st.set_page_config(
    page_title="🧠 Transformer Attention Visualizer",
    page_icon="🔍",
    layout="wide"
)

st.title("🧠 Transformer Attention Visualizer")

if not MODEL_LOADED:
    st.error(f"Model loading error: {ERROR_MSG}")
    st.info("Please wait while dependencies are being installed. This may take a few minutes on first deployment.")
    st.stop()

# Rest of your app code here...
