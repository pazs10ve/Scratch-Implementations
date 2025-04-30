import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class MultiQueryAttention(nn.Module):
    def __init__(self, embed_dim : torch.Tensor, num_heads : int, dropout : float=0.1) -> None:
        super(MultiQueryAttention, self).__init__()

        assert embed_dim % num_heads == 0, "Embedding dimensions should be divisible by NUM_HEADS"

        self.num_heads = num_heads

        # only query tensors are split into N heads

        self.query = nn.Linear(embed_dim, embed_dim)
        self.key = nn.Linear(embed_dim, embed_dim // num_heads)
        self.value = nn.Linear(embed_dim, embed_dim // num_heads)

        self.attention_dropout = nn.Dropout(dropout)

        self.out_proj = nn.Linear(embed_dim, embed_dim)
        self.proj_dropout = nn.Dropout(dropout)

    def forward(self, x : torch.Tensor, mask : torch.Tensor=None) -> torch.Tensor:

        """
        Args:
            x : Input tensor of size [BATCH_SIZE, SEQ_LEN, EMBED_DIM]
            mask : Tensor of size either [BATCH_SIZE, NUM_HEADS, SEQ_LEN, SEQ_LEN] or [1, NUM_HEADS, SEQ_LEN, SEQ_LEN]
        """

        BATCH_SIZE, SEQ_LEN, EMBED_DIM = x.size()

        Q = self.query(x)
        K = self.key(x)
        V = self.value(x)

        # split the query tensor into N heads
        Q = Q.reshape(BATCH_SIZE, SEQ_LEN, self.num_heads, EMBED_DIM // self.num_heads).permute(0, 2, 1, 3)

        attention_scores = torch.matmul(Q, K.unsqueeze(1).transpose(2, 3)) / math.sqrt(EMBED_DIM // self.num_heads)  # add extra dimension at position 1 to match the tensor shapes

        if mask is not None:
            attention_scores = attention_scores.masked_fill(mask == 0, float('-inf'))

        attention_weights = F.softmax(attention_scores, dim=-1)

        attention_weights = self.attention_dropout(attention_weights)

        attention_output = torch.matmul(attention_weights, V.unsqueeze(1)) # add extra dimension at position 1 to match the tensor shapes

        attention_output = attention_output.contiguous().view(BATCH_SIZE, SEQ_LEN, EMBED_DIM)

        out_proj = self.out_proj(attention_output)
        out = self.proj_dropout(out_proj)

        return out
    

"""
BATCH_SIZE = 6
SEQ_LEN = 32
EMBED_DIM = 768
NUM_HEADS = 16

mqa = MultiQueryAttention(embed_dim=EMBED_DIM, num_heads=NUM_HEADS)

x = torch.randn(BATCH_SIZE, SEQ_LEN, EMBED_DIM)

out = mqa(x)

print(out.shape) # [BATCH_SIZE, SEQ_LEN, EMBED_DIM]
"""