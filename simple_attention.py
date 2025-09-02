import torch
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from transformers import MarianMTModel, MarianTokenizer
import warnings
warnings.filterwarnings("ignore")

class SimpleAttentionVisualizer:
    def __init__(self, model_name="Helsinki-NLP/opus-mt-en-de"):
        """Simplified attention visualizer with just the essential views"""
        print("Loading model...")
        
        self.device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
        print(f"Using device: {self.device}")
        
        # Load model and tokenizer
        self.tokenizer = MarianTokenizer.from_pretrained(model_name)
        self.model = MarianMTModel.from_pretrained(model_name)
        self.model.to(self.device)
        self.model.eval()
        
        print("✅ Model loaded successfully!")
    
    def get_attention_data(self, source_text):
        """Extract attention weights and tokens"""
        
        # Tokenize and get translation
        inputs = self.tokenizer(source_text, return_tensors="pt", padding=True)
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        
        with torch.no_grad():
            # Generate translation
            outputs = self.model.generate(
                **inputs, 
                output_attentions=True,
                return_dict_in_generate=True,
                max_length=50,
                num_beams=1,
                do_sample=False
            )
            translation = self.tokenizer.decode(outputs.sequences[0], skip_special_tokens=True)
            
            # Get attention weights
            decoder_outputs = self.model(
                input_ids=inputs['input_ids'],
                attention_mask=inputs['attention_mask'],
                decoder_input_ids=outputs.sequences,
                output_attentions=True
            )
            attentions = decoder_outputs.cross_attentions
        
        # Get clean tokens
        source_tokens = self.tokenizer.convert_ids_to_tokens(inputs['input_ids'][0])
        target_tokens = self.tokenizer.convert_ids_to_tokens(outputs.sequences[0])
        
        # Clean tokens for display
        source_clean = [token.replace('▁', '').replace('</s>', '') for token in source_tokens if token not in ['<pad>']]
        target_clean = [token.replace('▁', '').replace('</s>', '') for token in target_tokens if token not in ['<pad>']]
        
        return attentions, source_clean, target_clean, translation
    
    def visualize_core_attention(self, source_text, layer=2, save_path=None):
        """
        THE essential view: Single attention heatmap showing word alignment
        This is the core of 'Attention is All You Need' - direct input-output connections
        """
        
        print(f"🎯 Analyzing: '{source_text}'")
        attentions, source_clean, target_clean, translation = self.get_attention_data(source_text)
        
        # Extract middle layer attention (usually most interpretable)
        if layer >= len(attentions):
            layer = len(attentions) - 1
            
        attention_matrix = attentions[layer][0, 0].cpu().numpy()  # First head
        
        # Adjust matrix size
        max_target = min(len(target_clean), attention_matrix.shape[0])
        max_source = min(len(source_clean), attention_matrix.shape[1])
        
        attention_clean = attention_matrix[:max_target, :max_source]
        source_display = source_clean[:max_source]
        target_display = target_clean[:max_target]
        
        # Create the visualization
        plt.figure(figsize=(max(10, len(source_display) * 1.2), max(8, len(target_display) * 0.8)))
        
        # Main heatmap
        sns.heatmap(
            attention_clean,
            xticklabels=source_display,
            yticklabels=target_display,
            cmap='Blues',
            annot=True,
            fmt='.2f',
            cbar_kws={'label': 'Attention Weight'},
            square=False,
            linewidths=0.5
        )
        
        plt.title(f'🧠 Transformer Attention: How Translation Happens\n'
                 f'English: "{source_text}"\n'
                 f'German: "{translation}"\n'
                 f'Layer {layer}, Head 0', fontsize=14, pad=20)
        
        plt.xlabel('🇺🇸 English Words (Source)', fontsize=12, fontweight='bold')
        plt.ylabel('🇩🇪 German Words (Target)', fontsize=12, fontweight='bold')
        
        # Rotate labels for better readability
        plt.xticks(rotation=45, ha='right')
        plt.yticks(rotation=0)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        plt.show()
        
        # Print interpretation
        print(f"\n📊 Translation: {translation}")
        print(f"🔍 How to read this heatmap:")
        print(f"   • Each row = one German word being generated")  
        print(f"   • Each column = one English word from input")
        print(f"   • Dark blue = high attention (model focusing here)")
        print(f"   • Light blue = low attention (model ignoring)")
        print(f"   • Look for diagonal patterns = direct word alignment")
        
        return translation
    
    def compare_heads(self, source_text, layer=2, save_path=None):
        """
        OPTIONAL: Compare 4 attention heads to see specialization
        Shows why multi-head attention is powerful
        """
        
        print(f"🔄 Comparing attention heads for: '{source_text}'")
        attentions, source_clean, target_clean, translation = self.get_attention_data(source_text)
        
        if layer >= len(attentions):
            layer = len(attentions) - 1
        
        # Create 2x2 subplot for first 4 heads
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        axes = axes.flatten()
        
        for head in range(min(4, attentions[layer].shape[1])):  # Show up to 4 heads
            attention_matrix = attentions[layer][0, head].cpu().numpy()
            
            # Adjust matrix size
            max_target = min(len(target_clean), attention_matrix.shape[0])
            max_source = min(len(source_clean), attention_matrix.shape[1])
            
            attention_clean = attention_matrix[:max_target, :max_source]
            
            sns.heatmap(
                attention_clean,
                xticklabels=source_clean[:max_source],
                yticklabels=target_clean[:max_target],
                cmap='Blues',
                ax=axes[head],
                cbar=True,
                annot=False,  # Too crowded with 4 subplots
                square=False
            )
            
            axes[head].set_title(f'Head {head}', fontsize=12, fontweight='bold')
            axes[head].set_xlabel('English Words')
            axes[head].set_ylabel('German Words')
            
        plt.suptitle(f'🧠 Multi-Head Attention Comparison (Layer {layer})\n'
                    f'English: "{source_text}" → German: "{translation}"', 
                    fontsize=14, y=0.98)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        plt.show()
        
        print(f"\n🔍 What you're seeing:")
        print(f"   • Each subplot = different attention head")
        print(f"   • Notice how heads show different patterns!")
        print(f"   • Some focus on word alignment, others on grammar")
        print(f"   • This proves why 'multi-head' attention works")
        
        return translation

# Quick test function
def quick_demo():
    """Run a quick demonstration"""
    viz = SimpleAttentionVisualizer()
    
    # Test sentences
    sentences = [
        "The cat sat on the mat.",
        "I love machine learning.",
        "Hello world!"
    ]
    
    print("\n🎯 Running simplified attention analysis...")
    
    for sentence in sentences:
        print(f"\n{'='*60}")
        
        # Core visualization - THE most important one
        viz.visualize_core_attention(sentence, layer=2)
        
        input("\nPress Enter to see multi-head comparison...")
        
        # Optional: Multi-head comparison
        viz.compare_heads(sentence, layer=2)
        
        input(f"\nPress Enter for next sentence...")

if __name__ == "__main__":
    quick_demo()
