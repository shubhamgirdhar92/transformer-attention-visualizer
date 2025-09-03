import streamlit as st
import warnings
warnings.filterwarnings("ignore")

# Import our visualizer class
from simple_attention import SimpleAttentionVisualizer

# Configure the page
st.set_page_config(
    page_title="🤔 Andaazaa Apna Apna",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Title and description
st.title("🤔 Andaazaa Apna Apna")

st.image("Andaazaa-poster.png", 
         caption="When AI goes W.A.I.L.D! 🤖", 
         use_container_width=True, width=600)

st.markdown("""
### Understanding Transformer "Andaazaa" (Approximation)
Andaazaa is an Urdu word that means approximation, which is essentially what transformer models do. See how they perform their approximation by visualizing attention weights.

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
st.sidebar.markdown(f"""
**Layer {layer} Focus:**
- **Layer 0-1**: Basic word alignment & position
- **Layer 2-3**: Grammar & syntax relationships  
- **Layer 4-5**: Full context & semantic meaning
""")
visualization_type = st.sidebar.radio(
    "Choose Visualization:",
    ["Single Head View", "Multi-Head Comparison"],
    help="Single head is clearer, multi-head shows specialization"
)

if visualization_type == "Multi-Head Comparison":
    st.sidebar.markdown("""
    **What Each Head Usually Does:**
    - **Head 0-1**: Word-to-word translation
    - **Head 2-3**: Grammar & sentence structure
    - **Head 4-5**: Context & positional info
    - **Head 6-7**: Semantic relationships
    
    Look for different patterns in each subplot!
    """)
else:
    st.sidebar.markdown("""
    **Single Head Analysis:**
    Shows one attention head's strategy.
    Try different layers to see how 
    attention evolves through the network.
    """)
analyze_button = st.sidebar.button("🔍 Analyze Attention", type="primary", use_container_width=True)

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
    if st.sidebar.button(f"📝 {sentence[:30]}..."):
        source_text = sentence
        st.rerun()

# Main content area
col1, col2 = st.columns([2, 1])

with col1:
    # Main visualization button
    if analyze_button:
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
                        translation = viz.visualize_core_attention(
                            source_text, 
                            layer=layer
                        )
                        
                        # Display the plot in Streamlit
                        st.pyplot(fig=None, clear_figure=True)
                        
                        # Show results
                        st.success(f"**Translation:** {translation}")
                        
                    else:
                        # Generate multi-head comparison
                        translation = viz.compare_heads(
                            source_text,
                            layer=layer
                        )
                        
                        # Display the plot
                        st.pyplot(fig=None, clear_figure=True)
                        
                        # Show results
                        st.success(f"**Translation:** {translation}")
                        
                    # Add interpretation guide
                    if visualization_type == "Multi-Head Comparison":
                        st.markdown("""
                        ### 📖 How to Read Multi-Head Comparison:
                        **Each subplot shows a different "expert" within the same layer:**
                        
                        **🎯 Head Specialization Patterns:**
                        - **Diagonal patterns**: Direct word alignment heads
                        - **Scattered attention**: Context and grammar heads  
                        - **Strong columns**: Heads focusing on specific word types
                        - **Broad distributions**: Semantic understanding heads
                        
                        **🔍 What to Compare:**
                        - Which heads show clear word-to-word mapping?
                        - Which heads attend to function words (the, is, on)?
                        - Which heads spread attention across multiple words?
                        - How do attention strategies differ between heads?
                        
                        **💡 Key Insight:** This proves why "multi-head" attention works - 
                        different heads learn different translation strategies!
                        """)
                    else:
                        st.markdown("""
                        ### 📖 How to Read Single Head View:
                        - **Rows (Y-axis):** German words being generated
                        - **Columns (X-axis):** English words from input
                        - **Dark blue:** High attention (model focusing here)
                        - **Light blue:** Low attention (model ignoring)
                        - **Diagonal patterns:** Direct word-to-word translation
                        - **Scattered patterns:** Contextual/grammatical relationships
                        
                        **💡 Try different layers to see how attention patterns evolve!**
                        """)
                    
            except Exception as e:
                st.error(f"Error: {str(e)}")
                st.error("Try a simpler sentence or check your internet connection.")

with col2:
    # Information panel
    st.markdown("""
    ### 🎓 What You're Learning:
    
    **"Andaazaa" (Approximation)** is how transformers work:
    
    ✅ **Pattern Recognition**: Models learn patterns from millions of examples
    
    ✅ **Attention Weights**: Show which words are important for translation
    
    ✅ **No Perfect Rules**: Just very good approximations based on data
    
    ✅ **Context Matters**: Same word can have different "Andaazaa" in different contexts
    
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
### 🚀 About "Andaazaa Apna Apna"
This visualizer demonstrates how transformer models make their "Andaazaa" (approximations) when translating. 
Just like humans make educated guesses based on context, transformers use attention to approximate the best translations!

**Model**: Helsinki-NLP/opus-mt-en-de (English to German translation)
**Architecture**: Transformer with 6 layers, 8 attention heads each

*Andaazaa (اندازہ) = Approximation/Estimate in Urdu/Hindi* 🎯
""")
