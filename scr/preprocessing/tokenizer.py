
import torch
class SimpleTokenizer:
    def __init__(self, text=None):
        if text:
            vocab = sorted(set(text))
            self.stoi = {ch: i for i, ch in enumerate(vocab)}
            self.itos = {i: ch for ch, i in self.stoi.items()}
            self.vocab_size = len(self.stoi)
        else:
            self.stoi = {}
            self.itos = {}
            self.vocab_size = 0

    def build_vocab(self, text):
        text.split()
        vocab = sorted(set(text))
        self.stoi = {ch: i for i, ch in enumerate(vocab)}
        self.itos = {i: ch for ch, i in self.stoi.items()}
        self.vocab_size = len(self.stoi)

    def encode(self, text):
        return torch.tensor([self.stoi[c] for c in text], dtype=torch.long)

    def decode(self, tokens):
        return ''.join([self.itos[int(i)] for i in tokens])