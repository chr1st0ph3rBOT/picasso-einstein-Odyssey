# view_logits.py
import torch
from torch import nn
import torch.nn.functional as F
from transformers import BertModel, BertLMHeadModel, BertTokenizer
import os
import matplotlib.pyplot as plt
import numpy as np

# --- Model Class Definition ---
class ArtistX(nn.Module):
    def __init__(self, model_name='bert-base-uncased'):
        super().__init__()
        self.tokenizer = BertTokenizer.from_pretrained(model_name)
        self.encoder = BertModel.from_pretrained(model_name)
        self.decoder = BertLMHeadModel.from_pretrained(model_name, is_decoder=True)

    def forward(self, input_ids, attention_mask):
        encoder_outputs = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
        latent_vector = encoder_outputs.last_hidden_state
        output_logits = self.decoder(inputs_embeds=latent_vector, attention_mask=attention_mask).logits
        return output_logits

def visualize_latent_vector(vector, token_name):
    """Visualizes the latent vector as a heatmap."""
    plt.figure(figsize=(12, 2))
    vector_2d = vector.cpu().numpy().reshape(1, -1)
    plt.imshow(vector_2d, cmap='coolwarm', aspect='auto')
    plt.colorbar(label='Activation Value')
    plt.title(f"Latent Vector for Token: '{token_name}'")
    plt.xlabel("Latent Dimensions (0-767)")
    plt.yticks([])
    plt.tight_layout()

def visualize_logits(tokens, logits, correct_token):
    """Visualizes the prediction logits as a bar chart."""
    plt.figure(figsize=(10, 8))
    y_pos = np.arange(len(tokens))
    
    colors = ['#3B82F6' if token == correct_token else '#9CA3AF' for token in tokens]
    
    bars = plt.barh(y_pos, logits, align='center', color=colors)
    plt.yticks(y_pos, tokens)
    plt.gca().invert_yaxis()
    plt.xlabel('Logit Value (Raw Score)')
    plt.title(f"Top 10 Predicted Tokens (Logits) for Position of '{correct_token}'")

    for bar in bars:
        width = bar.get_width()
        plt.text(width + 0.1, bar.get_y() + bar.get_height()/2,
                 f'{width:.2f}',
                 va='center')
    
    plt.tight_layout()

if __name__ == '__main__':
    # --- Settings ---
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    INPUT_TEXT = "This is a secret message."
    MODEL_PATH = "artist_x_best_model.pth"
    
    TOKEN_POSITION_TO_ANALYZE = 4 

    print("="*60)
    print("Artist X Decoder's Logits Analyzer")
    print("="*60)

    if not os.path.exists(MODEL_PATH):
        print(f"Error: Model file '{MODEL_PATH}' not found.")
        exit()

    # --- Model Loading ---
    print(f"\n[1] Loading Artist X model ('{MODEL_PATH}')...")
    model = ArtistX()
    model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
    model.to(DEVICE)
    model.eval()
    print(f"✅ Model loaded successfully. (Device: {str(DEVICE).upper()})")

    with torch.no_grad():
        # --- Encoding ---
        print(f"\n[2] Processing input text to generate latent vectors...")
        print(f"  - Input Text: '{INPUT_TEXT}'")
        
        inputs = model.tokenizer(INPUT_TEXT, return_tensors='pt', max_length=128, padding='max_length', truncation=True)
        inputs = {k: v.to(DEVICE) for k, v in inputs.items()}
        
        latent_vectors = model.encoder(inputs['input_ids'], attention_mask=inputs['attention_mask']).last_hidden_state
        
        print(f"✅ Latent vectors generated. Shape: {latent_vectors.shape}")

        # --- Latent Vector Analysis ---
        actual_token_id = inputs['input_ids'][0, TOKEN_POSITION_TO_ANALYZE].item()
        actual_token = model.tokenizer.convert_ids_to_tokens([actual_token_id])[0]
        
        print(f"\n[3] Analyzing the latent vector for the token '{actual_token}' at position {TOKEN_POSITION_TO_ANALYZE}...")
        
        target_latent_vector = latent_vectors[0, TOKEN_POSITION_TO_ANALYZE]
        print(f"  - Extracted vector's first 5 values: {target_latent_vector[:5].tolist()}")
        visualize_latent_vector(target_latent_vector, actual_token)

        # --- Decoding and Probability Analysis ---
        print(f"\n[4] Decoding latent vectors to get prediction probabilities...")
        output_logits = model.decoder(inputs_embeds=latent_vectors, attention_mask=inputs['attention_mask']).logits
        
        print(f"✅ Logits generated. Shape: {output_logits.shape}")
        
        print(f"\n[5] Analyzing prediction probabilities for position {TOKEN_POSITION_TO_ANALYZE}...")
        
        target_logits = output_logits[0, TOKEN_POSITION_TO_ANALYZE]
        
        # Get top 10 logits and their indices
        top_10_logits, top_10_indices = torch.topk(target_logits, 10)
        
        # Also get probabilities for terminal output
        probabilities = F.softmax(target_logits, dim=-1)
        top_10_probs = probabilities[top_10_indices]

        print(f"  - Actual token at this position: '{actual_token}' (ID: {actual_token_id})")

        # --- Terminal Output ---
        print("\n--- Top 10 Predictions (Terminal) ---")
        print(f"{'Rank':<5}{'Predicted Token':<15}{'Logit':<15}{'Probability':<20}")
        print("-"*60)
        
        top_tokens = []
        
        for i in range(10):
            token_id = top_10_indices[i].item()
            token = model.tokenizer.convert_ids_to_tokens([token_id])[0]
            logit_val = top_10_logits[i].item()
            prob = top_10_probs[i].item()
            top_tokens.append(token)
            
            is_correct = "<- (Correct!)" if token_id == actual_token_id else ""
            print(f"{i+1:<5}{token:<15}{logit_val:<15.2f}{prob:.2%} {is_correct}")
    
    print("\n" + "="*60)
    print("Analysis complete. Generating plots...")
    print("="*60)

    # --- Plot Visualization ---
    visualize_logits(top_tokens, top_10_logits.cpu().numpy(), actual_token)
    plt.show() # Show all generated plots
