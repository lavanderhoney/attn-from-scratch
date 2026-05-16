import torch
import torch.nn as nn
import torch.nn.functional as F
import math

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
            # ADDITIVE MODIFICATION
            # Since mask contains 0.0 (keep) and -1e9 (discard), we just ADD.
            attn_score = attn_score + mask
        
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
        # view() converts from (batch, seq_len, d_model) -> (batch, seq_len, n_heads, d_k)
        # then we transpose to keep batch and n_heads together as "batch dimension", and the mat-mul happens in the last two dim only
        Q = Q.view(batch_size, -1, self.n_heads, self.d_k).transpose(1, 2)
        K = K.view(batch_size, -1, self.n_heads, self.d_k).transpose(1, 2)
        V = V.view(batch_size, -1, self.n_heads, self.d_k).transpose(1, 2)
        
        self.attn_weights, output = MultiHeadAttention.attention(Q, K, V, mask, self.dropout)
        
        # concatenate heads using view
        output = output.transpose(1, 2).contiguous() # restore back to original memory format
        output = output.view(batch_size, -1, self.d_model) #flatten
        return self.w_o(output)
