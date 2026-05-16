import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import math
from typing import Optional

class Embeddings(nn.Module):
    def __init__(self, vocab_size:int, d_model:int):
        super().__init__()
        self.vocab_size = vocab_size
        self.d_model = d_model
        self.embedding = nn.Embedding(vocab_size, d_model)
        
    def forward(self, x) -> torch.Tensor :
        """
        Args:
            x: input word id tensor of (batch_size, seq_len)
        Returns:
            word embeddings of (batch_size, seq_len, d_model) 
        """
        return self.embedding(x) * np.sqrt(self.d_model) # will this work? i.e broadcasting?

class PositionalEncoding(nn.Module):
    def __init__(self, seq_len: int, d_model:int = 512 ) -> None:
        """
        Args:
            seq_len: the maximum sequence length the model can handle
            d_model: embedding dimensions of the model (512 from the vanilla arch)
        """
        super().__init__()
        self.seq_len = seq_len
        self.d_model = d_model
        
        # the position embedding matrix (to be filled in later)
        pe = torch.zeros(seq_len, d_model)
        
        # a tensor of shape [seq_len, 1] for the 'pos' indexes
        pos = torch.arange(0, self.seq_len).unsqueeze(1)
        
        # log-trick to compute the exponent in the divisor, for numeric stability. shape: (d_model/2, ) or, (1, d_model/2) when broadcasting
        # this is computed for half the dimensions, because as the formula, the even and odd indices have the same freq
        # this can be seen from the 2i and 2i+1 for selecting, but inside the freq, we're using 2i, so that makes two consecutive indices have the same freq
        div = torch.exp(
            torch.arange(0, d_model, 2).float() * (-math.log(10000.0)/d_model) 
        )
        
        # Fill in the PE matrix
        pe[:, 0::2] = torch.sin(pos * div) # broadcasting, then point-wise multiplication
        pe[:, 1::2] = torch.cos(pos * div)
        
        # Add a batch dimension: [1, max_len, d_model]
        # This allows broadcating when adding to word embeddings later.
        pe = pe.unsqueeze(0)
        
        self.register_buffer('pe', pe)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Embeddings of shape [batch_size, seq_len, d_model]
        
        Returns:
            Embeddings with positional information added.
        """
        # slice the pe only upto the sequence length of x (teacher forcing in decoder)
        return x + self.pe[:, :x.size(1), :]

class FeedForwardBlock(nn.Module):
    def __init__(self, d_model:int = 512, d_ff: int = 2048, dropout: float = 0.1) -> None:
        super().__init__()
        self.d_model = d_model
        self.d_ff = d_ff
        self.linear_1 = nn.Linear(d_model, d_ff)
        self.dropout_1 = nn.Dropout(dropout)
        self.linear_2 = nn.Linear(d_ff, d_model)

    def forward(self, x):
        x = self.dropout_1(F.relu(self.linear_1(x)))
        x = self.linear_2(x)
        return x

class SkipConnection(nn.Module):
    def __init__(self, d_model:int = 512, dropout:float = 0.1):
        super().__init__()
        self.norm = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x: torch.Tensor, sublayer: torch.Tensor):
        return x + self.dropout(sublayer(self.norm(x))) # Pre-LN is easier to train, as it avoids gradients exploding at the start, which could happen with post-LN
    