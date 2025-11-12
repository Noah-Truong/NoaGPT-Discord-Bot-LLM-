import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from model import SimpleLLM
from tokenizer import WordTokenizer  # Changed from BPETokenizer
from dataset import TextDataset
from config import Config
import os

def save_checkpoint(model, optimizer, epoch, loss, path, is_best=False):
    """Save model checkpoint"""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': loss,
    }
    torch.save(checkpoint, path)
    print(f"Checkpoint saved: {path}")
    
    if is_best:
        best_path = os.path.join(os.path.dirname(path), 'best_model.pth')
        torch.save(checkpoint, best_path)
        print(f"Best model saved: {best_path}")

def train_model(model, dataloader, epochs, lr, device, save_every=5):
    model = model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()
    
    best_loss = float('inf')
    
    for epoch in range(epochs):
        total_loss = 0
        for batch_idx, (inputs, targets) in enumerate(dataloader):
            inputs, targets = inputs.to(device), targets.to(device)
            
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs.view(-1, outputs.size(-1)), targets.view(-1))
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            
            if batch_idx % 50 == 0:
                print(f"Epoch {epoch+1}, Batch {batch_idx}, Loss: {loss.item():.4f}")
        
        avg_loss = total_loss / len(dataloader)
        print(f"Epoch {epoch+1} completed. Average Loss: {avg_loss:.4f}\n")
        
        # Save checkpoint every N epochs
        if (epoch + 1) % save_every == 0:
            checkpoint_path = f'checkpoints/model_epoch_{epoch+1}.pth'
            save_checkpoint(model, optimizer, epoch, avg_loss, checkpoint_path)
        
        # Save best model
        if avg_loss < best_loss:
            best_loss = avg_loss
            checkpoint_path = f'checkpoints/model_epoch_{epoch+1}.pth'
            save_checkpoint(model, optimizer, epoch, avg_loss, checkpoint_path, is_best=True)
    
    return model

if __name__ == "__main__":
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Training on {device}\n")
    
    # Load training data
    print("=" * 50)
    print("Loading training data...")
    print("=" * 50)
    
    data_file = Config.MOTHERLOAD_DATA_PATH
    if not os.path.exists(data_file):
        print(f"Error: {data_file} not found!")
        print("Please run clean_data.py and data_mixer.py first")
        exit(1)
    
    with open(data_file, 'r', encoding='utf-8') as f:
        training_text = f.read()
    
    print(f"Loaded {len(training_text):,} characters")
    print(f"Approximately {len(training_text.split()):,} words\n")
    
    # Initialize Word tokenizer
    print("=" * 50)
    print("Creating Word Tokenizer...")
    print("=" * 50)
    tokenizer = WordTokenizer(training_text, max_vocab_size=10000)  # Using WordTokenizer
    print()
    
    # Test tokenization
    test_text = "hello world, this is a test"
    encoded = tokenizer.encode(test_text)
    decoded = tokenizer.decode(encoded)
    print(f"Tokenization test:")
    print(f"  Original: {test_text}")
    print(f"  Encoded:  {encoded[:20]}...")
    print(f"  Decoded:  {decoded}")
    print()
    
    # Create dataset and dataloader
    print("=" * 50)
    print("Creating dataset...")
    print("=" * 50)
    dataset = TextDataset(training_text, tokenizer, max_length=Config.MAX_LENGTH)
    dataloader = DataLoader(dataset, batch_size=Config.BATCH_SIZE, shuffle=True)
    
    print(f"Dataset size: {len(dataset):,} examples")
    print(f"Batch size: {Config.BATCH_SIZE}")
    print(f"Total batches: {len(dataloader):,}\n")
    
    # Initialize model
    print("=" * 50)
    print("Initializing model...")
    print("=" * 50)
    model = SimpleLLM(
        vocab_size=tokenizer.vocab_size,
        d_model=Config.D_MODEL,
        nhead=Config.NHEAD,
        num_layers=Config.NUM_LAYERS,
        dim_feedforward=Config.DIM_FEEDFORWARD,
        max_seq_len=Config.MAX_SEQ_LEN
    )
    
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Total parameters: {total_params:,}")
    print()
    
    # Train
    print("=" * 50)
    print("Starting training...")
    print("=" * 50)
    print()
    trained_model = train_model(model, dataloader, Config.EPOCHS, Config.LEARNING_RATE, device)
    
    # Save model and tokenizer
    os.makedirs('checkpoints', exist_ok=True)
    torch.save(trained_model.state_dict(), Config.MODEL_SAVE_PATH)
    tokenizer.save(Config.TOKENIZER_SAVE_PATH)
    
    print("\n" + "=" * 50)
    print("Training complete! 🎉")
    print("=" * 50)
    print(f"Model saved to: {Config.MODEL_SAVE_PATH}")
    print(f"Tokenizer saved to: {Config.TOKENIZER_SAVE_PATH}")
    print(f"\nRun 'python generate.py' to test your model!")