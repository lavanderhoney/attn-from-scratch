import torch
import torch.nn as nn
import torch.nn.functional as F
from src.encoder import Encoder
from src.decoder import Decoder

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
            