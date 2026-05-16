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
    
class Transformer(nn.Module):
    def __init__(self, src_vocab_size: int, tgt_vocab_size:int, src_seq_len:int, tgt_seq_len: int,  pad_id:int, N: int = 6, 
                 d_model: int = 512, d_ff:int = 2048, n_heads: int = 8, dropout: float = 0.1):
        super().__init__()
        self.src_seq_len = src_seq_len
        self.tgt_seq_len = tgt_seq_len
        self.pad_id = pad_id
        self.encoder = Encoder(src_vocab_size, src_seq_len, N, d_model, d_ff, n_heads, dropout)
        self.decoder = Decoder(tgt_vocab_size, tgt_seq_len, N, d_model, d_ff, n_heads, dropout)
        self.projection_layer = nn.Linear(d_model, tgt_vocab_size)
        self._init_parameters()
    
    def forward(self, src, tgt, encoder_memory=None):
        # 1. Create Masks
        # (B, 1, 1, SrcLen) - Expanded for heads
        src_mask = self.make_src_mask(src, self.pad_id) # moved the reshaping to the function itself

        # (B, 1, TgtLen, TgtLen) - Expanded for heads
        tgt_mask = self.make_tgt_mask(tgt, self.pad_id)

        # 2. Encoder
        if torch.is_tensor(encoder_memory):
            # avoid calling the encoder again. used for inferencing
            memory = encoder_memory
        else:
            # Uses src_mask to ignore pads in self-attention
            memory = self.encoder(src, src_mask=src_mask)

        # 3. Decoder
        # tgt_mask: Used in Self-Attention (Mask Future + Pads)
        # src_mask: Used in Cross-Attention (Mask Encoder Pads)
        out = self.decoder(tgt, memory, src_mask=src_mask, tgt_mask=tgt_mask)
        out = self.projection_layer(out)
        return out # don't return probabilities, logits instead since CE loss is being used
   
    def _init_parameters(self):
        for p in self.parameters():
            if p.dim() > 1 :
                nn.init.xavier_uniform_(p)
                
    # masking. Use KEEP, additive mask
    # 1/True -> use this token, 0/False -> ignore this token, pad it
    def make_src_mask(self, src: torch.Tensor, pad_id:int):
        """
            Creates an additive mask for the Encoder (Source).
            src: (batch, seq_len)
            mask shape: (batch, 1, 1, seq_Len)
        """
        # create the boolean mask
        mask = (src != pad_id).unsqueeze(1).unsqueeze(2)
        
        # convert to additive float mask
        return torch.zeros_like(mask, dtype=torch.float).masked_fill(~mask, -1e9)
    
    def make_tgt_mask(self, tgt: torch.Tensor, pad_id: int):
        """
            Creates an additive causal mask for the Decoder (Target).
            Shape: (Batch, 1, Seq_Len, Seq_Len)
        """
        B, L = tgt.shape
        device = tgt.device
        
        # padding mask,(Keep non-pads) - (Batch, 1, 1, Seq_Len)
        pad_mask = (tgt != pad_id).unsqueeze(1).unsqueeze(2)
        
        # Causal Mask (Keep Lower Triangle) - (Seq_Len, Seq_Len)
        causal_mask = torch.tril(
            torch.ones((L, L), device=device)
        ).bool()
        
        # Combine: We keep a position if it is NOT padding AND it is in the Past/Current
        combined_mask = pad_mask & causal_mask
        
        return torch.zeros_like(combined_mask, dtype=torch.float).masked_fill(~combined_mask, -1e9)
        
    def greedy_decode(self, src: torch.Tensor, sos_id: int, eos_id: int) -> torch.Tensor:
        batch = src.size(0)
        device = src.device
        
        # src_mask = self.make_src_mask(src, self.pad_id).unsqueeze(1).unsqueeze(2) - moved the reshaping to the function itself
        src_mask = self.make_src_mask(src, self.pad_id)        
        memory = self.encoder(src, src_mask=src_mask) # (batch, seq_len, d_model)
        
        # start with just the SOS token
        decoder_input = torch.full((batch, 1), sos_id, dtype=torch.long, device=device)
        
        # Track which sequences have finished
        finished = torch.zeros(batch, dtype=torch.bool, device=device)

        for _ in range(self.tgt_seq_len - 1):
            tgt_mask = self.make_tgt_mask(decoder_input, self.pad_id)
            
            out = self.decoder(decoder_input, memory, src_mask=src_mask, tgt_mask=tgt_mask)
            prob = self.projection_layer(out) # (batch, seq_len, vocab_size)
            
            # 1. Isolate the prediction for the LAST timestep
            # prob[:, -1, :] has shape (batch, vocab_size)
            next_word_prob = prob[:, -1, :]
            
            # 2. Get the index of the max probability over the vocab dimension
            # next_tokens has shape (batch, )
            next_tokens = torch.argmax(next_word_prob, dim=-1)
            
            # Do not overwrite tokens after EOS. For postn finished is true, it will take a PAD id, otherwise next_token id 
            next_tokens = torch.where(
                finished,
                torch.full_like(next_tokens, self.pad_id),
                next_tokens
            )
            
            # Concatenate the new token
            decoder_input = torch.cat([decoder_input, next_tokens.unsqueeze(1)], dim=1)
            
            # update finished mask
            finished |= (next_tokens == eos_id)
            
            if finished.all():
                break
        return decoder_input
    
    def beam_search_decode(self, src: torch.Tensor, sos_id:int, eos_id:int, beam_width:int=2) -> torch.Tensor:
        """
        Initially, keep track of beam_width number of highest probab output tokens. For each timestamp, or sequence, again generate
        beam_width number of token with respective prev generated token in decoder_input. 

        Assuming src doesn't have batch dimension, only (1, seq_len)
        """
        device = src.device
        
        src_mask = self.make_src_mask(src, self.pad_id)        
        memory = self.encoder(src, src_mask=src_mask)
        print("memory: ", memory.shape)
        
        # initial forward pass
        decoder_input = torch.full((1, 1), fill_value=sos_id, dtype=torch.long, device=device)
        logits = self.forward(src, decoder_input, memory)
        
        log_probs = F.log_softmax(logits[:, -1, :], dim=-1)
        
        # get the initial candidates
        beam_scores, beam_tokens = torch.topk(log_probs, k=beam_width) # (1, beam_width)
        beam_scores = beam_scores.squeeze(0) # (k,)
        beam_tokens = beam_tokens.squeeze(0)
        
        # initialize beam state
        # final_seq = torch.full((beam_width, 1), fill_value=sos_id, dtype=torch.long, device=device) # will grow to (beam_width, tgt_seq_len)
        active_seq = torch.cat([
            torch.full((beam_width, 1), fill_value=sos_id, dtype=torch.long, device=device),
            beam_tokens.unsqueeze(1)
        ], dim = -1)
        
        # Expand encoder outputs to batch size 'k' for parallel cross-attention
        memory = memory.expand(beam_width, -1, -1) # (k, src_len, d_model)
        
        src = src.expand(beam_width, -1)
        
        # we already have the first token generated 
        for _ in range(self.tgt_seq_len - 2):
            # Forward pass for all active beams in parallel
            logits = self.forward(src, active_seq, memory) 
            
            next_log_probs = F.log_softmax(logits[:, -1, :], dim=-1) # (k, vocab_size)
            
            #Add the historical beam score to the new token scores
            candidate_scores = beam_scores.unsqueeze(1) + next_log_probs # (k,1) + (k,vocab_size), broadcasting
           
            # Flatten the scores to find the top k overall across ALL beams and vocabulary
            vocab_size = candidate_scores.size(-1)
            candidate_scores = candidate_scores.view(-1) # or use reshape ? 
            
            top_scores, top_indices = torch.topk(candidate_scores, k=beam_width)
            
            # track lineage
            beam_indices = torch.div(top_indices, vocab_size, rounding_mode='trunc')
            token_indices = top_indices % vocab_size
            
            # update sequence and scores
            beam_scores = top_scores
            # the wining parent beam indices, based on the new scores
            selected_indices = active_seq[beam_indices]
            
            #append the new tokens
            active_seq = torch.cat([
                selected_indices,
                token_indices.unsqueeze(1)
            ], dim=-1) # (k, current_seq_len + 1)
            
            if (token_indices == eos_id).all():
                break
        best_idx = torch.argmax(beam_scores)
        best_seq = active_seq[best_idx]
        return best_seq.unsqueeze(0)
            