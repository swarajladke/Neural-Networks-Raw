"""
Raw AGNIS — src/agnis/text/char_pipeline.py

Character-level data pipeline mapping text to one-hot transitions.
"""

import torch
from typing import List, Tuple, Iterator, Optional

class CharVocab:
    """Fixed character vocabulary with char<->index mapping."""
    
    def __init__(self, chars: Optional[List[str]] = None):
        """If chars is None, use the default Phase 4 character set."""
        if chars is None:
            # Build default: a-z, 0-9, space, newline, punctuation
            chars = sorted(list(set(
                list('abcdefghijklmnopqrstuvwxyz') +
                list('0123456789') +
                [' ', '\n'] +
                list('.,:;!?+-*=()')
            )))
        self.chars = sorted(list(set(chars)))
        self.char_to_idx = {c: i for i, c in enumerate(self.chars)}
        self.idx_to_char = {i: c for i, c in enumerate(self.chars)}
        self.vocab_size = len(self.chars)
    
    def encode(self, char: str) -> int:
        if char not in self.char_to_idx:
            # Fallback to space if character not in vocabulary to avoid crashes
            return self.char_to_idx.get(' ', 0)
        return self.char_to_idx[char]
    
    def decode(self, idx: int) -> str:
        return self.idx_to_char.get(idx, ' ')
    
    def to_onehot(self, char: str) -> torch.Tensor:
        vec = torch.zeros(self.vocab_size)
        vec[self.encode(char)] = 1.0
        return vec
    
    def from_onehot(self, vec: torch.Tensor) -> str:
        return self.decode(vec.argmax().item())
    
    def encode_string(self, text: str) -> List[int]:
        return [self.encode(c) for c in text]
    
    def decode_indices(self, indices: List[int]) -> str:
        return ''.join(self.decode(i) for i in indices)


class CharacterStream:
    """Yields (x_onehot, y_onehot, x_idx, y_idx) character transition pairs from text."""
    
    def __init__(self, text: str, vocab: CharVocab):
        self.text = text
        self.vocab = vocab
        self.indices = vocab.encode_string(text)
    
    def __len__(self):
        return max(0, len(self.indices) - 1)
    
    def __iter__(self) -> Iterator[Tuple[torch.Tensor, torch.Tensor, int, int]]:
        for i in range(len(self.indices) - 1):
            x_idx = self.indices[i]
            y_idx = self.indices[i + 1]
            
            x_oh = torch.zeros(self.vocab.vocab_size)
            x_oh[x_idx] = 1.0
            
            y_oh = torch.zeros(self.vocab.vocab_size)
            y_oh[y_idx] = 1.0
            
            yield x_oh, y_oh, x_idx, y_idx
