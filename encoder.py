import torch
import torch.nn as nn
from typing import Optional
from layers import Embeddings, PositionalEncoding, SkipConnection, FeedForwardBlock
from attention import MultiHeadAttention

class EncoderBlock(nn.Module):
    def __init__(self, d_model: int=512, n_heads: int =8, d_ff: int = 2048, dropout:float =0.1):
        super().__init__()
        self.mha_attn_block = MultiHeadAttention(d_model, n_heads, dropout)
        self.ff_block = FeedForwardBlock(d_model, d_ff, dropout)
        self.skip_connections = nn.ModuleList(
            [SkipConnection(d_model, dropout) for _ in range(2)]
        )
        
    def forward(self, x: torch.Tensor,  mask: Optional[torch.Tensor]):
        x = self.skip_connections[0](x, lambda x: self.mha_attn_block(x, x, x, mask))
        return self.skip_connections[1](x, self.ff_block)

class Encoder(nn.Module):
    def __init__(self,  vocab_size: int, seq_len:int, N: int = 6,  d_model: int = 512, d_ff:int = 2048, n_heads: int = 8, dropout: float = 0.1):
        super().__init__()
        self.embedding = Embeddings(vocab_size, d_model)
        self.pos_embeddings = PositionalEncoding(seq_len)
        self.encoder_blocks = nn.ModuleList(
            [EncoderBlock(d_model, n_heads, d_ff, dropout) for _ in range(N)]
        )
        self.norm = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x, src_mask=None):
        x = self.embedding(x)
        x = self.pos_embeddings(x)
        x = self.dropout(x)
        for encoder_block in self.encoder_blocks:
            x = encoder_block(x, src_mask)
        
        return self.norm(x)
    