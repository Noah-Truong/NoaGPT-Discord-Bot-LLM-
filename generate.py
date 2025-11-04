import torch
import torch.nn.functional as F
from model import SimpleLLM
from tokenizer import WordTokenizer # Changed from CharTokenizer
from config import Config

def generate_text(model, tokenizer, start_text, max_length=50, temperature=0.8, device='cpu'):
    """Generate text from prompt"""
    model.eval()
    tokens = tokenizer.encode(start_text)
    
    if len(tokens) == 0:
        tokens = [1]  # Start with <UNK> if empty
    
    generated_tokens = []
    
    with torch.no_grad():
        for _ in range(max_length):
            # Limit context window
            context = tokens[-Config.MAX_SEQ_LEN:]
            input_tensor = torch.tensor([context]).to(device)
            output = model(input_tensor)
            
            # Get next token
            logits = output[0, -1, :] / temperature
            probs = F.softmax(logits, dim=0)
            next_token = torch.multinomial(probs, 1).item()
            
            # Stop at end of sentence
            next_word = tokenizer.idx_to_token.get(next_token, '<UNK>')
            if next_word in ['.', '!', '?'] and len(generated_tokens) > 5:
                tokens.append(next_token)
                break
            
            tokens.append(next_token)
            generated_tokens.append(next_token)
    
    return tokenizer.decode(tokens)

if __name__ == "__main__":
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    # Load tokenizer
    print("Loading tokenizer...")
    tokenizer = WordTokenizer.load(Config.TOKENIZER_SAVE_PATH)
    
    # Load model
    print("Loading model...")
    model = SimpleLLM(
        vocab_size=tokenizer.vocab_size,
        d_model=Config.D_MODEL,
        nhead=Config.NHEAD,
        num_layers=Config.NUM_LAYERS,
        dim_feedforward=Config.DIM_FEEDFORWARD,
        max_seq_len=Config.MAX_SEQ_LEN
    )
    
    # Load checkpoint (contains model_state_dict, optimizer, etc.)
    checkpoint = torch.load('checkpoints/best_model.pth', map_location=device)
    
    # Extract just the model weights
    if 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
    else:
        model.load_state_dict(checkpoint)  # Fallback for old format
    
    model = model.to(device)
    model.eval()
    print("✅ Model loaded\n")
    
    # Interactive generation
    print("=" * 50)
    print("Text Generation (type 'quit' to exit)")
    print("=" * 50)
    print("\nTips:")
    print("- Start with a few words like 'hello' or 'i think'")
    print("- Try different temperatures (0.5 = safe, 1.0 = creative)")
    print()
    
    while True:
        prompt = input("\nEnter prompt: ").strip()
        if prompt.lower() == 'quit':
            break
        
        if not prompt:
            prompt = "hello"
        
        # Ask for temperature
        temp_input = input("Temperature (0.5-1.5, default 0.8): ").strip()
        try:
            temperature = float(temp_input) if temp_input else 0.8
        except:
            temperature = 0.8
        
        # Generate
        print("\nGenerating...")
        generated = generate_text(model, tokenizer, prompt, max_length=100, temperature=temperature, device=device)
        
        print("\n" + "=" * 50)
        print("Generated text:")
        print("=" * 50)
        print(generated)
        print("=" * 50)