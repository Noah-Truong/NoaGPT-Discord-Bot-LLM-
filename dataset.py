import torch
from torch.utils.data import Dataset

class TextDataset(Dataset):
    def __init__(self, text_data, tokenizer, max_length=128):
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.examples = []
        
        tokens = tokenizer.encode(text_data)
        for i in range(0, len(tokens) - max_length, max_length):
            self.examples.append(tokens[i:i + max_length])
    
    def __len__(self):
        return len(self.examples)
    
    def __getitem__(self, idx):
        tokens = self.examples[idx]
        return torch.tensor(tokens[:-1]), torch.tensor(tokens[1:])