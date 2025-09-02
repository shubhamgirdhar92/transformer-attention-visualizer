import streamlit as st
import warnings
warnings.filterwarnings("ignore")

# Import our visualizer class
from simple_attention import SimpleAttentionVisualizer

# Configure the page
st.set_page_config(
    page_title="🎯 Andaza Apna Apna",
    page_icon="🎯",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Hide Streamlit warnings with custom CSS
hide_warnings_css = """
<style>
/* Hide deprecation warnings */
.stAlert > div[data-baseweb="notification"] {
    display: none !important;
}

/* Hide specific warning text */
div[data-testid="stMarkdownContainer"] p:contains("deprecated") {
    display: none !important;
}

/* Hide any yellow warning boxes */
.stAlert[data-baseweb="notification"][kind="warning"] {
    display: none !important;
}

/* Hide matplotlib warnings */
.stAlert:contains("st.pyplot") {
    display: none !important;
}

/* General warning suppression */
[class*="warning"] {
    display: none !important;
}
</style>
"""
st.markdown(hide_warnings_css, unsafe_allow_html=True)

# Title and description
st.title("🎯 Andaza Apna Apna")

# Add the fun image right below the title
st.image("andaza-poster.png", 
         caption="When AI goes W.A.I.L.D! 🤖", 
         use_container_width=True)

st.markdown("""
### Understanding Transformer "Andaza" (Approximation)
Andaza is an urdu word which means approximation which is basically what transformer models do, see how to do their andaza/approximation by visualising attention weights!

**How to use:**
1. Enter an English sentence
2. Choose a layer and head to visualize
3. Click "Analyze Attention" to see the magic! ✨
""")

# Create sidebar controls
st.sidebar.header("🎛️ Controls")

# Input text
source_text = st.sidebar.text_input(
    "Enter English text:",
    value="The cat sat on the mat.",
    max_chars=100,  # Limit for web performance
    help="Keep sentences under 15 words for best performance"
)

# Validate input length
if len(source_text.split()) > 15:
    st.sidebar.warning("⚠️ Please use sentences under 15 words for better performance on web")

# Model controls
layer = st.sidebar.selectbox(
    "Select Layer:",
    options=[0, 1, 2, 3, 4, 5],
    index=2,
    help="Middle layers (2-3) often show the clearest patterns"
)

visualization_type = st.sidebar.radio(
    "Choose Visualization:",
    ["Single Head View", "Multi-Head Comparison"],
    help="Single head is clearer, multi-head shows specialization"
)

# Add examples
st.sidebar.markdown("### 💡 Try These Examples:")
example_sentences = [
    "The cat sat on the mat.",
    "I love machine learning.",
    "Hello world!",
    "The quick brown fox jumps.",
    "How are you today?"
]

for sentence in example_sentences:
    if st.sidebar.button(f"📝 {sentence[:20]}..."):
        source_text = sentence
        st.rerun()

# Main content area
col1, col2 = st.columns([2, 1])

with col1:
    # Main visualization button
    if st.button("🔍 Analyze Attention", type="primary", use_container_width=True):
        if not source_text.strip():
            st.error("Please enter some English text to analyze!")
        elif len(source_text.split()) > 15:
            st.error("Please use sentences under 15 words for better web performance!")
        else:
            try:
                # Show loading spinner
                with st.spinner("🤖 Loading model and generating visualization..."):
                    
                    # Initialize visualizer with caching
                    @st.cache_resource
                    def load_visualizer():
                        return SimpleAttentionVisualizer()
                    
                    viz = load_visualizer()
                    
                    if visualization_type == "Single Head View":
                        # Generate single head visualization
                        # Capture ALL output including Streamlit warnings
                        stderr_buffer = io.StringIO()
                        with redirect_stderr(stderr_buffer):
                            with warnings.catch_warnings():
                                warnings.simplefilter("ignore")
                                translation = viz.visualize_core_attention(
                                    source_text, 
                                    layer=layer
                                )
                        
                        # Display the plot without any warning output
                        try:
                            # Force matplotlib to not display warnings in Streamlit
                            import matplotlib
                            matplotlib.pyplot.ioff()  # Turn off interactive mode
                            st.pyplot(clear_figure=True, use_container_width=True)
                        except:
                            st.pyplot()
                        
                        # Show results
                        st.success(f"**Translation:** {translation}")
                        
                    else:
                        # Generate multi-head comparison
                        stderr_buffer = io.StringIO()
                        with redirect_stderr(stderr_buffer):
                            with warnings.catch_warnings():
                                warnings.simplefilter("ignore")
                                translation = viz.compare_heads(
                                    source_text,
                                    layer=layer
                                )
                        
                        # Display the plot without warnings
                        try:
                            import matplotlib
                            matplotlib.pyplot.ioff()
                            st.pyplot(clear_figure=True, use_container_width=True)
                        except:
                            st.pyplot()
                        
                        # Show results
                        st.success(f"**Translation:** {translation}")
                        
                    # Add interpretation guide
                    st.markdown("""
                    ### 📖 How to Read the Visualization:
                    - **Rows (Y-axis):** German words being generated
                    - **Columns (X-axis):** English words from input
                    - **Dark blue:** High attention (model focusing here)
                    - **Light blue:** Low attention (model ignoring)
                    - **Diagonal patterns:** Direct word-to-word translation
                    - **Scattered patterns:** Contextual/grammatical relationships
                    """)
                    
            except Exception as e:
                st.error(f"Error: {str(e)}")
                st.error("Try a simpler sentence or check your internet connection.")

with col2:
    # Information panel
    st.markdown("""
    ### 🎓 What You're Learning:
    
    **"Andaza" (Approximation)** is how transformers work:
    
    ✅ **Pattern Recognition**: Models learn patterns from millions of examples
    
    ✅ **Attention Weights**: Show which words are important for translation
    
    ✅ **No Perfect Rules**: Just very good approximations based on data
    
    ✅ **Context Matters**: Same word can have different "andaza" in different contexts
    
    ### 🔍 What to Look For:
    - Strong attention (0.7+) = confident approximation
    - Distributed attention = considering multiple possibilities
    - Different heads = different approximation strategies
    """)
    
    # Add some examples of what to expect
    st.markdown("""
    ### 🎯 Expected Patterns:
    - **"cat" → "Katze"**: High direct attention
    - **"the" → "Die"**: May attend to multiple words for grammar
    - **Verbs**: Often attend to subjects for agreement
    """)

# Footer
st.markdown("---")
st.markdown("""
### 🚀 About "Andaza Apna Apna"
This visualizer demonstrates how transformer models make their "andaza" (approximations) when translating. 
Just like humans make educated guesses based on context, transformers use attention to approximate the best translations!

**Model**: Helsinki-NLP/opus-mt-en-de (English to German translation)
**Architecture**: Transformer with 6 layers, 8 attention heads each

*Andaza (اندازہ) = Approximation/Estimate in Urdu/Hindi* 🎯
""")
