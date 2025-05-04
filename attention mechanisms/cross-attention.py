import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class CrossAttention(nn.Module):
    def __init__(self, in_embed_dim : int, out_embed_dim : int, num_heads : int, dropout : float=0.1) -> None:
        super(CrossAttention, self).__init__()
        
        assert (in_embed_dim % num_heads == 0) and (out_embed_dim % num_heads == 0), "Stream embedding dimensions should be divisible by NUM_HEADS"
        self.num_heads = num_heads

        self.query = nn.Linear(in_embed_dim, out_embed_dim, bias=False)
        self.key = nn.Linear(out_embed_dim, out_embed_dim, bias=False)
        self.value = nn.Linear(out_embed_dim, out_embed_dim, bias=False)

        self.attention_dropout = nn.Dropout(dropout)

        self.out_proj = nn.Linear(out_embed_dim, out_embed_dim, bias=False)
        self.proj_dropout = nn.Dropout(dropout)


    def forward(self, x1 : torch.Tensor, x2 : torch.Tensor, mask=None) -> torch.Tensor:
        """
        Args:
            x1: Source tensor of shape [BATCH_SIZE, SEQ_LEN_A, EMBED_DIM_A]
            x2: Target tensor of shape [BATCH_SIZE, SEQ_LEN_B, EMBED_DIM_B]
        """

        batch_size, seq_len_a, embed_dim_a = x1.size()
        _, seq_len_b, embed_dim_b = x2.size()

        Q = self.query(x1)
        K = self.key(x2)
        V = self.value(x2)

        # split the input kqv tensors into N heads

        Q = Q.reshape(batch_size, seq_len_a, self.num_heads, embed_dim_b // self.num_heads).permute(0, 2, 1, 3)
        K = K.reshape(batch_size, seq_len_b, self.num_heads, embed_dim_b // self.num_heads).permute(0, 2, 1, 3)
        V = V.reshape(batch_size, seq_len_b, self.num_heads, embed_dim_b // self.num_heads).permute(0, 2, 1, 3)

        # calculate the attention scores between the query from the source and key from the target sequence 
        attention_scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(embed_dim_b // self.num_heads)

        # apply the attention mask if present
        if mask is not None:
            attention_scores = attention_scores.masked_fill(mask == 0, float('-inf'))

        # apply softmax to the attention scores
        attention_weights = F.softmax(attention_scores, dim=-1)

        # calculate the weighted attention scores wrt the target sequence
        attention_output = torch.matmul(attention_weights, V)

        # reshape the attention tensors to the target sequence format
        attention_output = attention_output.contiguous().view(batch_size, seq_len_a, embed_dim_b)
        attention_output = self.attention_dropout(attention_output)

        # project the attention output into the target sequence 
        out_proj = self.out_proj(attention_output)
        out = self.proj_dropout(out_proj)

        return out





"""
BATCH_SIZE = 6
SEQ_LEN_A = 14
SEQ_LEN_B = 20
EMBED_DIM_A = 128
EMBED_DIM_B = 256
NUM_HEADS = 8

x1 = torch.randn(BATCH_SIZE, SEQ_LEN_A, EMBED_DIM_A)
x2 = torch.randn(BATCH_SIZE, SEQ_LEN_B, EMBED_DIM_B)

attention = CrossAttention(in_embed_dim=EMBED_DIM_A, out_embed_dim=EMBED_DIM_B, num_heads=NUM_HEADS)
out = attention(x1, x2)

print(out.shape) # [BATCH_SIZE, SEQ_LEN_A, EMBED_DIM_B] == torch.Size([6, 14, 256])
"""