import re
from collections import Counter, defaultdict

class BPETokenizer:
    """Byte Pair Encoding tokenizer - learns subword units"""
    
    def __init__(self, text, vocab_size=5000, min_frequency=2):
        self.vocab_size = vocab_size
        self.min_frequency = min_frequency
        
        # Special tokens
        self.special_tokens = ['<PAD>', '<UNK>', '<START>', '<END>']
        
        print("Building BPE tokenizer...")
        print(f"Target vocab size: {vocab_size}")
        
        # Step 1: Initialize with characters
        words = self._get_words(text)
        self.word_freq = Counter(words)
        
        # Start with character-level vocabulary
        vocab = set()
        for word in self.word_freq.keys():
            vocab.update(word)
        
        # Add special tokens
        self.vocab = self.special_tokens + sorted(list(vocab))
        
        # Step 2: Learn BPE merges
        print("Learning subword merges...")
        self.merges = self._learn_bpe(words, vocab_size - len(self.vocab))
        
        # Step 3: Build final vocabulary
        self._build_vocab()
        
        print(f"Final vocab size: {len(self.vocab)}")
        print(f"Number of merges: {len(self.merges)}")
    
    def _get_words(self, text):
        """Extract words and add end-of-word marker"""
        # Split on whitespace and add </w> marker
        words = re.findall(r'\S+', text.lower())
        # Add space between characters for BPE
        return [' '.join(list(word)) + ' </w>' for word in words]
    
    def _get_stats(self, words):
        """Count frequency of adjacent pairs"""
        pairs = defaultdict(int)
        for word, freq in words.items():
            symbols = word.split()
            for i in range(len(symbols) - 1):
                pairs[symbols[i], symbols[i + 1]] += freq
        return pairs
    
    def _merge_vocab(self, pair, words):
        """Merge most frequent pair in vocabulary"""
        new_words = {}
        bigram = ' '.join(pair)
        replacement = ''.join(pair)
        
        for word, freq in words.items():
            new_word = word.replace(bigram, replacement)
            new_words[new_word] = freq
        
        return new_words
    
    def _learn_bpe(self, words, num_merges):
        """Learn BPE merges"""
        words_dict = {word: self.word_freq[word.replace(' ', '').replace('</w>', '')] 
                      for word in words}
        
        merges = []
        for i in range(num_merges):
            pairs = self._get_stats(words_dict)
            if not pairs:
                break
            
            best_pair = max(pairs, key=pairs.get)
            
            # Only merge if frequent enough
            if pairs[best_pair] < self.min_frequency:
                break
            
            words_dict = self._merge_vocab(best_pair, words_dict)
            merges.append(best_pair)
            
            if (i + 1) % 100 == 0:
                print(f"  Learned {i + 1} merges...")
        
        return merges
    
    def _build_vocab(self):
        """Build final vocabulary from merges"""
        vocab_set = set(self.vocab)
        
        # Add all merged tokens
        for pair in self.merges:
            merged = ''.join(pair)
            vocab_set.add(merged)
        
        self.vocab = sorted(list(vocab_set))
        self.token_to_idx = {token: i for i, token in enumerate(self.vocab)}
        self.idx_to_token = {i: token for token, i in self.token_to_idx.items()}
        self.vocab_size = len(self.vocab)
    
    def _apply_bpe(self, word):
        """Apply BPE merges to a word"""
        # Add space between characters
        word = ' '.join(list(word)) + ' </w>'
        
        # Apply merges in order
        for pair in self.merges:
            bigram = ' '.join(pair)
            replacement = ''.join(pair)
            word = word.replace(bigram, replacement)
        
        return word.split()
    
    def encode(self, text):
        """Encode text to token IDs"""
        words = re.findall(r'\S+', text.lower())
        
        token_ids = []
        for word in words:
            subwords = self._apply_bpe(word)
            for subword in subwords:
                token_id = self.token_to_idx.get(subword, 1)  # 1 = <UNK>
                token_ids.append(token_id)
        
        return token_ids
    
    def decode(self, token_ids):
        """Decode token IDs to text"""
        tokens = [self.idx_to_token.get(idx, '<UNK>') for idx in token_ids]
        
        # Remove special tokens
        tokens = [t for t in tokens if t not in self.special_tokens]
        
        # Reconstruct text
        text = ''.join(tokens)
        
        # Remove </w> markers and add spaces
        text = text.replace('</w>', ' ')
        
        # Clean up spacing
        text = ' '.join(text.split())
        
        return text
    
    def save(self, path):
        import pickle
        with open(path, 'wb') as f:
            pickle.dump(self, f)
    
    @staticmethod
    def load(path):
        import pickle
        with open(path, 'rb') as f:
            return pickle.load(f)


class WordTokenizer:
    """Simple word-level tokenizer (fallback)"""
    def __init__(self, text, max_vocab_size=10000):
        tokens = re.findall(r'\w+|[^\w\s]', text.lower())
        
        from collections import Counter
        token_counts = Counter(tokens)
        most_common = token_counts.most_common(max_vocab_size - 2)
        
        self.vocab = ['<PAD>', '<UNK>'] + [token for token, _ in most_common]
        self.token_to_idx = {token: i for i, token in enumerate(self.vocab)}
        self.idx_to_token = {i: token for token, i in self.token_to_idx.items()}
        self.vocab_size = len(self.vocab)
        
        print(f"Vocabulary size: {self.vocab_size}")
    
    def encode(self, text):
        tokens = re.findall(r'\w+|[^\w\s]', text.lower())
        return [self.token_to_idx.get(token, 1) for token in tokens]
    
    def decode(self, token_ids):
        tokens = [self.idx_to_token.get(idx, '<UNK>') for idx in token_ids]
        
        result = []
        for i, token in enumerate(tokens):
            if token in ['<PAD>', '<UNK>']:
                continue
            
            if i > 0 and token not in '.,!?;:\'")-]}>':
                result.append(' ')
            
            result.append(token)
        
        return ''.join(result)
    
    def save(self, path):
        import pickle
        with open(path, 'wb') as f:
            pickle.dump(self, f)
    
    @staticmethod
    def load(path):
        import pickle
        with open(path, 'rb') as f:
            return pickle.load(f)


class CharTokenizer:
    """Character-level tokenizer (backup)"""
    def __init__(self, text):
        self.chars = sorted(list(set(text)))
        self.char_to_idx = {ch: i for i, ch in enumerate(self.chars)}
        self.idx_to_char = {i: ch for i, ch in enumerate(self.chars)}
        self.vocab_size = len(self.chars)
    
    def encode(self, text):
        return [self.char_to_idx[ch] for ch in text if ch in self.char_to_idx]
    
    def decode(self, tokens):
        return ''.join([self.idx_to_char[t] for t in tokens])
    
    def save(self, path):
        import pickle
        with open(path, 'wb') as f:
            pickle.dump(self, f)
    
    @staticmethod
    def load(path):
        import pickle
        with open(path, 'rb') as f:
            return pickle.load(f)