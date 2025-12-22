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
    def __init__(self,seq_len: int, d_model:int = 512 ) -> None:
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
        # slice the pe only upto the sequence length of x
        return x + self.pe[:, :x.size(1)+1, :]

class MultiHeadAttention(nn.Module):
    def __init__(self, d_model: int = 512, n_heads: int = 8, dropout: float = 0.1):
        super().__init__()
        assert d_model % n_heads == 0, "d_model must be divisible by n_heads"
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_k = d_model // n_heads     

        self.w_q = nn.Linear(d_model, d_model)
        self.w_k = nn.Linear(d_model, d_model)
        self.w_v = nn.Linear(d_model, d_model)
        
        self.w_o = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout)
        
    @staticmethod
    def attention(Q, K, V, mask, dropout: nn.Module):
        """
        Performs the attention function. 
        Returns:
            attn_weights: a (batch, n_heads, seq, seq) tensor of self-attention weights.
            
            output: a (batch, n_heads, seq, d_k) tensor, the final ouput of self-attention.
        """
        d_k = Q.shape[-1]
        attn_score = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(d_k)
        
        if mask is not None:
            attn_score.masked_fill(mask==0, 1e-9)
        
        attn_weights = F.softmax(attn_score, dim=-1)
        attn_weights = dropout(attn_weights)
        
        output = torch.matmul(attn_weights, V)  # (batch, n_heads, seq, seq) x (batch, n_heads, seq, d_k) -> (batch, n_heads, seq, d_k)
        return attn_weights, output
        
    def forward(self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, mask=None):
        """
        Args:
            q, k, v: Input tensors of shape (Batch_Size, Seq_Len, d_model)
                     For Self-Attention, pass the same tensor for all three.
            mask: Optional tensor of shape (Batch_Size, 1, 1, Seq_Len) or (Batch, 1, Seq, Seq)
        """
        batch_size = q.shape[0]
        # Linear projections
        Q = self.w_q(q)
        K = self.w_k(k)
        V = self.w_v(v)
        
        # Split into heads
        # view() converts from (batch, seq_len, d_k) -> (batch, seq_len, n_heads, d_k)
        # then we transpose to keep batch and n_heads together as "batch dimension", and the mat-mul happens in the last two dim only
        Q = Q.view(batch_size, -1, self.n_heads, self.d_k).transpose(1, 2)
        K = K.view(batch_size, -1, self.n_heads, self.d_k).transpose(1, 2)
        V = V.view(batch_size, -1, self.n_heads, self.d_k).transpose(1, 2)
        
        self.attn_weights, output = MultiHeadAttention.attention(Q, K, V, mask, self.dropout)
        
        # concatenate heads using view
        output = output.transpose(1, 2).contiguous() # restore back to original memory format
        output = output.view(batch_size, -1, self.d_model) #flatten
        return self.w_o(output)

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
        return x + self.dropout(self.norm(sublayer)) # Pre-LN is easier to train, as it avoids gradients exploding at the start, which could happen with post-LN
    
class EncoderBlock(nn.Module):
    def __init__(self, d_model: int=512, n_heads: int =8, d_ff: int = 2048, dropout:float =0.1):
        super().__init__()
        self.mha_attn_block = MultiHeadAttention(d_model, n_heads, dropout)
        self.ff_block = FeedForwardBlock(d_model, d_ff, dropout)
        self.skip_connections = nn.ModuleList(
            [SkipConnection(d_model, dropout) for _ in range(2)]
        )
        
    def forward(self, x: torch.Tensor,  mask: Optional[torch.Tensor]):
        attn_out = self.mha_attn_block(x, x, x, mask)
        x = self.skip_connections[0](x, attn_out)        
        ff_out = self.ff_block(x)
        return self.skip_connections[1](x, ff_out)

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
        _x = self.self_attn(x, x, x, tgt_mask)
        x = self.skip_connections[0](x, _x)
        
        # 2. Cross-Attention (Query=x, Key=Encoder, Value=Encoder)
        # Uses src_mask (Source Padding)
        _x = self.cross_attn(q=x, k=encoder_output, v=encoder_output, mask=src_mask)
        x = self.skip_connections[1](x, _x)
        
        # 3. Feed-forward
        _x = self.ff_block(x)
        x = self.skip_connections[2](x, _x)
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
        
    def forward(self, x, encoder_output, src_mask, target_mask):
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
            x = decoder_block(x, encoder_output, src_mask, target_mask)
        return self.norm(x)
    
class Transformer(nn.Module):
    def __init__(self, src_vocab_size: int, tgt_vocab_size:int, src_seq_len:int, tgt_seq_len: int,  N: int = 6, 
                 d_model: int = 512, d_ff:int = 2048, n_heads: int = 8, dropout: float = 0.1):
        super().__init__()
        self.src_seq_len = src_seq_len
        self.tgt_seq_len = tgt_seq_len
        self.encoder = Encoder(src_vocab_size, src_seq_len, N, d_model, d_ff, n_heads, dropout)
        self.decoder = Decoder(tgt_vocab_size, tgt_seq_len, N, d_model, d_ff, n_heads, dropout)
        self.projection_layer = nn.Linear(d_model, tgt_vocab_size)
        self._init_parameters()
    
    def forward(self, src, tgt):
        # 1. Create Masks
        # (B, 1, 1, SrcLen) - Expanded for heads
        src_mask = self.make_src_pad_mask(src, self.pad_id).unsqueeze(1).unsqueeze(2) 

        # (B, 1, TgtLen, TgtLen) - Expanded for heads
        tgt_mask = self.make_decoder_mask(tgt, self.pad_id).unsqueeze(1)

        # 2. Encoder
        # Uses src_mask to ignore pads in self-attention
        memory = self.encoder(src, mask=src_mask)

        # 3. Decoder
        # tgt_mask: Used in Self-Attention (Mask Future + Pads)
        # src_mask: Used in Cross-Attention (Mask Encoder Pads)
        out = self.decoder(tgt, memory, tgt_mask=tgt_mask, src_mask=src_mask)
        out = self.projection_layer(out)
        return out # don't return probabilities, CE loss is being used
   
    def _init_parameters(self):
        for p in self.parameters():
            if p.dim() > 1 :
                nn.init.xavier_uniform_(p)
                
    # masking
    def make_src_pad_mask(self, src_ids, pad_id):
        # src_ids = (B, seq_len)
        return (src_ids == pad_id)
    
    def make_tgt_pad_mask(self, tgt_ids, pad_id):
        return (tgt_ids == pad_id)   # (B, T)

    def make_causal_mask(self, size, device):
        return torch.triu(
            torch.ones(size, size, dtype=torch.bool, device=device),
            diagonal=1
        )
    def make_decoder_mask(self, tgt_ids, pad_id):
        B, T = tgt_ids.shape

        pad_mask = self.make_tgt_pad_mask(tgt_ids, pad_id)     # (B, T)
        causal = self.make_causal_mask(T, tgt_ids.device)      # (T, T)

        # broadcast to (B, T, T)
        return pad_mask.unsqueeze(1) | causal

# class Encoder(nn.Module):
#     def __init__(self, N: int = 6, d_model: int = 512, n_heads: int = 8, droput: float = 0.1):
#         self.N = N
#         self.d_model = d_model
#         self.h = n_heads
#         self.encoder_block = nn.Sequential(
#             *[MultiHeadAttention(d_model, n_heads, droput) for _ in range(N)]
#         )
# class MultiHeadAttention(nn.Module):
#     def __init__(self, d_model:int=512, n_heads:int=4) -> None:
#         super().__init__()
#         self.d_model = d_model
#         self.h = n_heads
#         self.concat_layer = nn.Linear(d_model, d_model)
#     def forward(self, x: torch.Tensor) -> torch.Tensor :
#         """
        
#         Args:
#             x: input tensor of shape: (batch_size, seq_len, d_model)
#         inp embedding (x) -> ( q,k,v proj layer -> (q, k, v) -> attn layer ) x h -> concat -> linear
#         """
#         heads = []
#         for head in range(self.h):
#             attn = Attention(x)
#             heads.append(attn)
#         head_tensor = torch.hstack(tuple(heads))
#         return self.concat_layer(head_tensor)   
# class Attention(nn.Module):
#     """
#     Class for computing the scaled dot-product self-attention function.
#     Will be used as the base class, which the MHA will wrap around
#     """
#     def __init__(self, attn_mask: Optional[torch.Tensor]=None, d_model:int=512, n_heads:int=6) -> None:
    
#         super().__init__()

#         self.mask = attn_mask
#         self.d_model = d_model
#         self.h = n_heads
#         self.q_proj = nn.Linear(self.d_model, self.d_model//self.h)
#         self.k_proj = nn.Linear(self.d_model, self.d_model//self.h)
#         self.v_proj = nn.Linear(self.d_model, self.d_model//self.h)
#         self.attention_weights = None
        
#     def forward(self, x: torch.Tensor):
#         q_d = self.q_proj(x)
#         k_d = self.k_proj(x)
#         v_d = self.v_proj(x)
#         attn_score = torch.matmul(q_d, k_d.transpose(0, 1))
        
#         if self.mask is not None:
#             pass
        
#         self.attention_weights = F.softmax(attn_score/math.sqrt(self.d_model))
#         return torch.matmul(self.attention_weights, v_d)
