import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class SlidingWindowAttention(nn.Module):
    def __init__(self, embed_dim : int, num_heads : int, window_length : int, dropout : float=0.1) -> None:
        super(SlidingWindowAttention, self).__init__()

        """
        Args:
            embed_dim (int): Dimensions of the input embeddings.
            num_heads (int): Number of attention heads.
            window_length (int): Span of attention length on either side.
            dropout (float): Dropout probability of attention scores.  
        """

        assert embed_dim % num_heads == 0, "Embeddings dimensions must be divisble by NUM_HEADS"

        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.window_length = window_length
        self.head_dim = self.embed_dim // self.num_heads

        self.query = nn.Linear(embed_dim, embed_dim)
        self.key = nn.Linear(embed_dim, embed_dim)
        self.value = nn.Linear(embed_dim, embed_dim)

        self.attention_dropout = nn.Dropout(dropout)

        self.out_proj = nn.Linear(embed_dim, embed_dim)
        self.proj_dropout = nn.Dropout(dropout)


    def forward(self, x : torch.Tensor, mask=None) -> torch.Tensor:

        BATCH_SIZE, SEQ_LEN, EMBED_DIM = x.size()

        query = self.query(x)
        key = self.key(x)
        value = self.value(x)

        query = query.reshape(BATCH_SIZE, SEQ_LEN, self.num_heads, self.head_dim).transpose(1, 2)
        key = key.reshape(BATCH_SIZE, SEQ_LEN, self.num_heads, self.head_dim).transpose(1, 2)
        value = value.reshape(BATCH_SIZE, SEQ_LEN, self.num_heads, self.head_dim).transpose(1, 2)

        # create the output mask 
        attention_output = torch.zeros_like(x)

        # iterate over the token length to compute local attention
        for i in range(SEQ_LEN):

            start_idx = max(0, i-self.window_length)
            end_idx = min(SEQ_LEN, i+self.window_length+1)

            # extract the local kqv pairs to compute the attention scores over
            query_i = query[:, :, i:i+1, :]
            k_window = key[:, :, start_idx:end_idx, :]
            v_window = value[:, :, start_idx:end_idx, :]

            attention_score = torch.matmul(query_i, k_window.transpose(-1, -2)) / math.sqrt(self.head_dim)

            if mask is not None:
                mask_window = mask[:, start_idx:end_idx].unsqueeze(1).unsqueeze(2)
                attention_score = attention_score.masked_fill(mask_window==0, float('-inf'))

            attention_weights = F.softmax(attention_score, dim=-1)

            attention_weights = self.attention_dropout(attention_weights)

            attention_window = torch.matmul(attention_weights, v_window).transpose(1, 2)
            attention_window = attention_window.reshape(BATCH_SIZE, 1, EMBED_DIM)

            # populate the output mask 
            attention_output[:, i:i+1, :] = attention_window 
        
        out_proj = self.out_proj(attention_output)
        
        out = self.proj_dropout(out_proj)

        return out

        

"""
EMBED_DIM = 768
NUM_HEADS = 12
SEQ_LEN = 64
WINDOW_LENGTH = 24
BATCH_SIZE = 32

x = torch.randn(BATCH_SIZE, SEQ_LEN, EMBED_DIM)

swa = SlidingWindowAttention(embed_dim=EMBED_DIM, num_heads=NUM_HEADS, window_length=WINDOW_LENGTH)
out = swa(x)
print(out.shape)
"""
            

            










