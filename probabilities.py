import torch
import torch.nn.functional as F
from model import SimpleLLM
from tokenizer import WordTokenizer
from config import Config

def get_token_probabilities(model, tokenizer, text, device='cpu', top_k=10):
    """
    Tokenize text and return probabilities for each predicted token
    
    Args:
        model: Trained SimpleLLM model
        tokenizer: CharTokenizer
        text: Input text to analyze
        device: 'cpu' or 'cuda'
        top_k: Number of top predictions to show for each position
    
    Returns:
        List of dictionaries with token info and probabilities
    """
    model.eval()
    tokens = tokenizer.encode(text)
    
    results = []
    
    with torch.no_grad():
        for i in range(len(tokens) - 1):
            # Get context up to current position
            context = tokens[:i+1]
            input_tensor = torch.tensor([context]).to(device)
            
            # Get model predictions
            output = model(input_tensor)
            logits = output[0, -1, :]  # Last token's predictions
            probs = F.softmax(logits, dim=0)
            
            # Get top k predictions
            top_probs, top_indices = torch.topk(probs, top_k)
            
            # Actual next token
            actual_token = tokens[i + 1]
            actual_char = tokenizer.decode([actual_token])
            actual_prob = probs[actual_token].item()
            
            # Top predictions
            top_predictions = [
                {
                    'token': tokenizer.decode([idx.item()]),
                    'probability': prob.item()
                }
                for prob, idx in zip(top_probs, top_indices)
            ]
            
            results.append({
                'position': i,
                'context': tokenizer.decode(context),
                'actual_next_char': actual_char,
                'actual_probability': actual_prob,
                'top_predictions': top_predictions
            })
    
    return results

def analyze_text(model, tokenizer, text, device='cpu'):
    """
    Analyze text and show detailed probability breakdown
    """
    print("=" * 80)
    print(f"Analyzing: '{text}'")
    print("=" * 80 + "\n")
    
    results = get_token_probabilities(model, tokenizer, text, device, top_k=5)
    
    for result in results:
        print(f"Position {result['position']}:")
        print(f"  Context: '{result['context']}'")
        print(f"  Actual next char: '{result['actual_next_char']}' (probability: {result['actual_probability']:.4f})")
        print(f"  Top 5 predictions:")
        
        for i, pred in enumerate(result['top_predictions'], 1):
            char = pred['token'].replace('\n', '\\n').replace('\t', '\\t')
            prob = pred['probability']
            bar = '█' * int(prob * 50)  # Visual bar
            print(f"    {i}. '{char}' - {prob:.4f} {bar}")
        
        print()

def get_full_sequence_probability(model, tokenizer, text, device='cpu'):
    """
    Calculate the overall probability of the entire sequence
    """
    model.eval()
    tokens = tokenizer.encode(text)
    
    total_log_prob = 0.0
    
    with torch.no_grad():
        for i in range(len(tokens) - 1):
            context = tokens[:i+1]
            input_tensor = torch.tensor([context]).to(device)
            
            output = model(input_tensor)
            logits = output[0, -1, :]
            log_probs = F.log_softmax(logits, dim=0)
            
            actual_token = tokens[i + 1]
            total_log_prob += log_probs[actual_token].item()
    
    # Convert log probability to regular probability
    avg_log_prob = total_log_prob / (len(tokens) - 1)
    perplexity = torch.exp(torch.tensor(-avg_log_prob)).item()
    
    return {
        'total_log_probability': total_log_prob,
        'average_log_probability': avg_log_prob,
        'perplexity': perplexity,
        'num_tokens': len(tokens)
    }

def compare_texts(model, tokenizer, texts, device='cpu'):
    """
    Compare probabilities of multiple text sequences
    """
    print("=" * 80)
    print("Comparing text sequences:")
    print("=" * 80 + "\n")
    
    results = []
    for text in texts:
        metrics = get_full_sequence_probability(model, tokenizer, text, device)
        results.append((text, metrics))
    
    # Sort by perplexity (lower is better)
    results.sort(key=lambda x: x[1]['perplexity'])
    
    for i, (text, metrics) in enumerate(results, 1):
        print(f"{i}. \"{text}\"")
        print(f"   Average log probability: {metrics['average_log_probability']:.4f}")
        print(f"   Perplexity: {metrics['perplexity']:.2f}")
        print(f"   Total tokens: {metrics['num_tokens']}")
        print()

if __name__ == "__main__":
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}\n")
    
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
    
    checkpoint = torch.load(Config.MODEL_SAVE_PATH, map_location=device)
    if 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
    else:
        model.load_state_dict(checkpoint)
    
    model = model.to(device)
    model.eval()
    print("✅ Model loaded\n")
    
    # Example 1: Analyze a single text
    test_text = "Hello world"
    analyze_text(model, tokenizer, test_text, device)
    
    # Example 2: Get overall probability
    print("\n" + "=" * 80)
    print("Overall Sequence Probability:")
    print("=" * 80 + "\n")
    metrics = get_full_sequence_probability(model, tokenizer, test_text, device)
    print(f"Text: '{test_text}'")
    print(f"Total log probability: {metrics['total_log_probability']:.4f}")
    print(f"Average log probability: {metrics['average_log_probability']:.4f}")
    print(f"Perplexity: {metrics['perplexity']:.2f}")
    print(f"Number of tokens: {metrics['num_tokens']}")
    
    # Example 3: Compare multiple texts
    print("\n")
    compare_texts(model, tokenizer, [
        "Hello world",
        "xyz abc qwe",
        "How are you"
    ], device)
    
    # Interactive mode
    print("\n" + "=" * 80)
    print("Interactive Mode (type 'quit' to exit)")
    print("=" * 80 + "\n")
    
    while True:
        user_input = input("Enter text to analyze: ").strip()
        if user_input.lower() == 'quit':
            break
        
        if user_input:
            print()
            analyze_text(model, tokenizer, user_input, device)
            metrics = get_full_sequence_probability(model, tokenizer, user_input, device)
            print(f"Perplexity: {metrics['perplexity']:.2f}\n")