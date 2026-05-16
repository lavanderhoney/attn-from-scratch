import torch
import torch.nn as nn
from typing import Optional
from layers import Embeddings, PositionalEncoding, SkipConnection, FeedForwardBlock
from attention import MultiHeadAttention

class DecoderBlock(nn.Module):
    def __init__(self, d_model: int=512, n_heads: int =8, d_ff: int = 2048, dropout:float =0.1) -> None:
        super().__init__()
        
        # We need TWO separate attention layers
        self.self_attn = MultiHeadAttention(d_model, n_heads, dropout)
        self.cross_attn = MultiHeadAttention(d_model, n_heads, dropout)
        
        self.ff_block = FeedForwardBlock(d_model, d_ff, dropout)
        self.skip_connections = nn.ModuleList(
            [SkipConnection(d_model, dropout) for _ in range(3)]
        )
    def forward(self, x: torch.Tensor, encoder_output: torch.Tensor, src_mask: torch.Tensor, tgt_mask: torch.Tensor):
        # 1. Masked Self-Attention (Query=x, Key=x, Value=x)
        # Uses tgt_mask (Look-Ahead + Padding)
        x = self.skip_connections[0](x, lambda x: self.self_attn(x, x, x, tgt_mask))
        
        # 2. Cross-Attention (Query=x, Key=Encoder, Value=Encoder)
        # Uses src_mask (Source Padding)
        x = self.skip_connections[1](x, lambda x: self.cross_attn(q=x, k=encoder_output, v=encoder_output, mask=src_mask))
        
        # 3. Feed-forward
        x = self.skip_connections[2](x, self.ff_block)
        return x

class Decoder(nn.Module):
    def __init__(self, vocab_size: int, seq_len:int, N: int = 6, d_model: int = 512, 
                 d_ff:int = 2048, n_heads: int = 8, dropout: float = 0.1):
        super().__init__()
        self.embedding = Embeddings(vocab_size, d_model)
        self.pos_embeddings = PositionalEncoding(seq_len)
        self.decoder_blocks = nn.ModuleList(
            [DecoderBlock(d_model, n_heads, d_ff, dropout) for _ in range(N)]
        )
        self.norm = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x, encoder_output, src_mask, tgt_mask):
        """
        Args:
            x: Decoder inputs (Target IDs)
            encoder_output: The output from the Encoder
            src_mask: Mask to hide Encoder padding from Cross-Attn
            tgt_mask: Mask to hide Future tokens from Self-Attn
        """
        x = self.embedding(x)
        x = self.pos_embeddings(x)
        x = self.dropout(x)
        for decoder_block in self.decoder_blocks:
            x = decoder_block(x, encoder_output, src_mask, tgt_mask)
        return self.norm(x)
    