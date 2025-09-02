# Import necessary libraries
import torch                    # PyTorch for tensor operations
import matplotlib.pyplot as plt # For creating plots
import seaborn as sns          # For beautiful heatmaps
import numpy as np             # For numerical operations
from transformers import MarianMTModel, MarianTokenizer  # Hugging Face transformer models
import warnings
warnings.filterwarnings("ignore")  # Hide warning messages for cleaner output

class SimpleAttentionVisualizer:
    def __init__(self, model_name="Helsinki-NLP/opus-mt-en-de"):
        """
        Initialize the attention visualizer
        
        Args:
            model_name: Pre-trained translation model (English to German)
        """
        print("Loading model...")
        
        # Check if Apple Silicon GPU (MPS) is available, otherwise use CPU
        self.device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
        print(f"Using device: {self.device}")
        
        # Load the tokenizer (converts text to numbers that model understands)
        self.tokenizer = MarianTokenizer.from_pretrained(model_name)
        
        # Load the translation model with eager attention (needed for visualization)
        self.model = MarianMTModel.from_pretrained(model_name, attn_implementation="eager")
        
        # Move model to GPU/CPU and set to evaluation mode (no training)
        self.model.to(self.device)
        self.model.eval()
        
        print("✅ Model loaded successfully!")
    
    def get_attention_data(self, source_text):
        """
        Extract attention weights and tokens from the model
        
        This function:
        1. Converts text to numbers (tokenization)
        2. Runs the model to get translation + attention weights
        3. Converts numbers back to readable tokens
        4. Returns everything needed for visualization
        
        Args:
            source_text: English sentence to translate
            
        Returns:
            attentions: Raw attention weights from all layers
            source_clean: Clean English tokens for display
            target_clean: Clean German tokens for display  
            translation: The German translation
        """
        
        # STEP 1: Convert English text to numbers the model understands
        inputs = self.tokenizer(source_text, return_tensors="pt", padding=True)
        inputs = {k: v.to(self.device) for k, v in inputs.items()}  # Move to GPU/CPU
        
        # STEP 2: Run the model (no gradients needed since we're not training)
        with torch.no_grad():
            # Generate German translation with attention tracking
            outputs = self.model.generate(
                **inputs,                      # Input English tokens
                output_attentions=True,        # CRITICAL: Save attention weights!
                return_dict_in_generate=True,  # Return structured output
                max_length=50,                 # Don't generate super long sentences
                num_beams=1,                   # Use greedy decoding (not beam search)
                do_sample=False                # Don't use random sampling
            )
            
            # Convert German number tokens back to text
            translation = self.tokenizer.decode(outputs.sequences[0], skip_special_tokens=True)
            
            # STEP 3: Get the actual attention weights
            # We need to run the model again to get cross-attention between English and German
            decoder_outputs = self.model(
                input_ids=inputs['input_ids'],           # English tokens
                attention_mask=inputs['attention_mask'], # Which tokens to pay attention to
                decoder_input_ids=outputs.sequences,    # Generated German tokens
                output_attentions=True                   # Save the attention weights!
            )
            attentions = decoder_outputs.cross_attentions  # These are the attention weights!
        
        # STEP 4: Convert token numbers back to readable words
        source_tokens = self.tokenizer.convert_ids_to_tokens(inputs['input_ids'][0])
        target_tokens = self.tokenizer.convert_ids_to_tokens(outputs.sequences[0])
        
        # STEP 5: Clean up tokens for better display
        # Remove special symbols like ▁ (subword marker) and <pad> (padding)
        source_clean = [token.replace('▁', '').replace('</s>', '') for token in source_tokens if token not in ['<pad>']]
        target_clean = [token.replace('▁', '').replace('</s>', '') for token in target_tokens if token not in ['<pad>']]
        
        return attentions, source_clean, target_clean, translation
    
    def visualize_core_attention(self, source_text, layer=2, save_path=None):
        """
        Create the MAIN attention heatmap - this shows the core of how transformers work!
        
        This visualization answers: "Which English words does the model look at 
        when generating each German word?"
        
        Args:
            source_text: English sentence to analyze
            layer: Which transformer layer to visualize (0-5, middle layers often most interpretable)
            save_path: Optional path to save the plot
            
        Returns:
            translation: The German translation produced
        """
        
        print(f"🎯 Analyzing: '{source_text}'")
        
        # STEP 1: Get all the data we need
        attentions, source_clean, target_clean, translation = self.get_attention_data(source_text)
        
        # STEP 2: Extract attention weights for specific layer
        # Make sure we don't ask for a layer that doesn't exist
        if layer >= len(attentions):
            layer = len(attentions) - 1
            
        # Extract attention matrix: [batch_size, num_heads, target_length, source_length]
        # We take [0, 0] = first batch, first attention head
        attention_matrix = attentions[layer][0, 0].cpu().numpy()  # Move from GPU to CPU for plotting
        
        # STEP 3: Make sure tokens and attention matrix sizes match
        # Sometimes model generates extra tokens or has padding
        max_target = min(len(target_clean), attention_matrix.shape[0])  # Rows (German words)
        max_source = min(len(source_clean), attention_matrix.shape[1])  # Columns (English words)
        
        # Trim everything to matching sizes
        attention_clean = attention_matrix[:max_target, :max_source]
        source_display = source_clean[:max_source]
        target_display = target_clean[:max_target]
        
        # STEP 4: Create the visualization
        # Make figure size proportional to number of words (for better readability)
        fig, ax = plt.subplots(figsize=(max(10, len(source_display) * 1.2), max(8, len(target_display) * 0.8)))
        
        # Create the heatmap - this is the magic!
        sns.heatmap(
            attention_clean,                    # The attention weights (our main data!)
            xticklabels=source_display,         # English words on X-axis
            yticklabels=target_display,         # German words on Y-axis  
            cmap='Blues',                       # Color scheme (white to dark blue)
            annot=True,                         # Show actual numbers in each cell
            fmt='.2f',                          # Format numbers to 2 decimal places
            cbar_kws={'label': 'Attention Weight'},  # Label for color bar
            square=False,                       # Don't force square cells
            linewidths=0.5,                     # Add thin lines between cells
            ax=ax                               # Plot on our specific axes
        )
        
        # STEP 5: Add informative title and labels
        ax.set_title(f'🧠 Transformer Attention: How Translation Happens\\n'
                    f'English: "{source_text}"\\n'
                    f'German: "{translation}"\\n'
                    f'Layer {layer}, Head 0', fontsize=14, pad=20)
        
        ax.set_xlabel('🇺🇸 English Words (Source)', fontsize=12, fontweight='bold')
        ax.set_ylabel('🇩🇪 German Words (Target)', fontsize=12, fontweight='bold')
        
        # Rotate labels for better readability
        plt.setp(ax.get_xticklabels(), rotation=45, ha='right')  # English words at 45° angle
        plt.setp(ax.get_yticklabels(), rotation=0)               # German words horizontal
        
        plt.tight_layout()  # Adjust spacing so everything fits nicely
        
        # STEP 6: Save if requested
        if save_path:
            fig.savefig(save_path, dpi=300, bbox_inches='tight')
        
        # Store translation for access by Streamlit app
        self.last_translation = translation
        
        # STEP 7: Print interpretation guide (for command line use)
        print(f"\\n📊 Translation: {translation}")
        print(f"🔍 How to read this heatmap:")
        print(f"   • Each row = one German word being generated")  
        print(f"   • Each column = one English word from input")
        print(f"   • Dark blue = high attention (model focusing here)")
        print(f"   • Light blue = low attention (model ignoring)")
        print(f"   • Look for diagonal patterns = direct word alignment")
        
        # Return the figure object for Streamlit
        return fig
    
    def compare_heads(self, source_text, layer=2, save_path=None):
        """
        Compare 4 attention heads to see specialization
        
        This shows WHY multi-head attention is powerful - each head learns 
        to focus on different aspects of the translation task.
        
        Args:
            source_text: English sentence to analyze
            layer: Which transformer layer to visualize 
            save_path: Optional path to save the plot
            
        Returns:
            translation: The German translation produced
        """
        
        print(f"🔄 Comparing attention heads for: '{source_text}'")
        
        # STEP 1: Get the attention data (same as before)
        attentions, source_clean, target_clean, translation = self.get_attention_data(source_text)
        
        # Make sure layer exists
        if layer >= len(attentions):
            layer = len(attentions) - 1
        
        # STEP 2: Create a 2x2 grid to show 4 different attention heads
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))  # 2 rows, 2 columns
        axes = axes.flatten()  # Convert 2D array to 1D for easier indexing
        
        # STEP 3: Plot each attention head separately
        for head in range(min(4, attentions[layer].shape[1])):  # Show up to 4 heads
            
            # Extract attention matrix for this specific head
            # attentions[layer] shape: [batch_size, num_heads, target_length, source_length]
            attention_matrix = attentions[layer][0, head].cpu().numpy()  # [0, head] = first batch, specific head
            
            # Make sure tokens and attention matrix sizes match (same as before)
            max_target = min(len(target_clean), attention_matrix.shape[0])
            max_source = min(len(source_clean), attention_matrix.shape[1])
            
            attention_clean = attention_matrix[:max_target, :max_source]
            
            # Create heatmap for this head
            sns.heatmap(
                attention_clean,
                xticklabels=source_clean[:max_source],  # English words
                yticklabels=target_clean[:max_target],  # German words
                cmap='Blues',                           # Same color scheme
                ax=axes[head],                          # Plot on specific subplot
                cbar=True,                              # Show color bar for each
                annot=False,                            # No numbers (too crowded with 4 plots)
                square=False                            # Don't force square cells
            )
            
            # Add title for this specific head
            axes[head].set_title(f'Head {head}', fontsize=12, fontweight='bold')
            axes[head].set_xlabel('English Words')
            axes[head].set_ylabel('German Words')
            
        # STEP 4: Add overall title
        fig.suptitle(f'🧠 Multi-Head Attention Comparison (Layer {layer})\\n'
                    f'English: "{source_text}" → German: "{translation}"', 
                    fontsize=14, y=0.98)
        
        plt.tight_layout()  # Adjust spacing
        
        # Save if requested
        if save_path:
            fig.savefig(save_path, dpi=300, bbox_inches='tight')
        
        # Store translation for access by Streamlit app
        self.last_translation = translation
        
        # STEP 5: Print explanation of what to look for (for command line use)
        print(f"\\n🔍 What you're seeing:")
        print(f"   • Each subplot = different attention head")
        print(f"   • Notice how heads show different patterns!")
        print(f"   • Some focus on word alignment, others on grammar")
        print(f"   • This proves why 'multi-head' attention works")
        
        # Return the figure object for Streamlit
        return fig

# DEMO FUNCTION - Shows how to use the visualizer
def quick_demo():
    """
    Run a quick demonstration of the attention visualizer
    
    This function:
    1. Creates a visualizer instance
    2. Tests it on 3 different sentences  
    3. Shows both single head and multi-head visualizations
    """
    
    # STEP 1: Create the visualizer (this loads the model)
    viz = SimpleAttentionVisualizer()
    
    # STEP 2: Test sentences with different complexity levels
    sentences = [
        "The cat sat on the mat.",        # Simple, clear word alignment expected
        "I love machine learning.",      # Technical terms, interesting to see
        "Hello world!"                   # Very simple, good baseline
    ]
    
    print("\\n🎯 Running simplified attention analysis...")
    
    # STEP 3: Analyze each sentence
    for sentence in sentences:
        print(f"\\n{'='*60}")
        
        # Show the CORE visualization - this is the most important one!
        # This shows which English words the model looks at for each German word
        viz.visualize_core_attention(sentence, layer=2)
        
        # Pause so user can examine the plot
        input("\\nPress Enter to see multi-head comparison...")
        
        # OPTIONAL: Show how different attention heads specialize
        # This proves why multi-head attention is powerful
        viz.compare_heads(sentence, layer=2)
        
        # Pause before next sentence
        input(f"\\nPress Enter for next sentence...")

# MAIN EXECUTION - This runs when you execute the script directly
if __name__ == "__main__":
    """
    This code block only runs when you execute this file directly
    (not when you import it as a module)
    
    It starts the demo automatically
    """
    quick_demo()
