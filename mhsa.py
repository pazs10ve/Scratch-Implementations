import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class MultiHeadSelfAttention(nn.Module):
    def __init__(self, embed_dim : int, num_heads : int, dropout : float=0.1) -> None:
        super(MultiHeadSelfAttention, self).__init__()

        assert embed_dim % num_heads == 0, "Embedding dimensions must be divisible by number of heads"

        self.num_heads = num_heads

        self.query = nn.Linear(embed_dim, embed_dim, bias=False)
        self.key = nn.Linear(embed_dim, embed_dim, bias=False)
        self.value = nn.Linear(embed_dim, embed_dim, bias=False)

        self.attention_dropout = nn.Dropout(dropout)

        self.out_proj = nn.Linear(embed_dim, embed_dim, bias=False)
        self.proj_dropout = nn.Dropout(dropout)


    def forward(self, x : torch.Tensor, mask : torch.Tensor=None) -> torch.Tensor:
        """
        Args: 
            x: Input tensor of shape [BATCH_SIZE, SEQ_LEN, EMBED_DIM]
            mask: Optional mask tensor of shape either [BATCH_SIZE, NUM_HEADS, SEQ_LEN, SEQ_LEN] or [1, NUM_HEADS, SEQ_LEN, SEQ_LEN]
        """

        BATCH_SIZE, SEQ_LEN, EMBED_DIM = x.size()

        Q = self.query(x)
        K = self.key(x)
        V = self.value(x)

        Q = Q.reshape(BATCH_SIZE, self.num_heads, SEQ_LEN, EMBED_DIM // self.num_heads)
        K = K.reshape(BATCH_SIZE, self.num_heads, SEQ_LEN, EMBED_DIM // self.num_heads)
        V = V.reshape(BATCH_SIZE, self.num_heads, SEQ_LEN, EMBED_DIM // self.num_heads)


        attention_scores = torch.matmul(Q, K.transpose(2, 3)) / math.sqrt(EMBED_DIM // self.num_heads)


        if mask is not None:
            attention_scores = attention_scores.masked_fill(mask == 0, float('-inf'))

        attention_weights = F.softmax(attention_scores, dim = -1)

        output = torch.matmul(attention_weights, V)
        output = output.transpose(1, 2).contiguous().view(BATCH_SIZE, SEQ_LEN, -1)



        output_proj = self.out_proj(output)
        output_proj = self.proj_dropout(output_proj)

        return output_proj
    


"""
BATCH_SIZE = 6
SEQ_LEN = 10
EMBED_DIM = 128
NUM_HEADS = 8

mhsa = MultiHeadSelfAttention(embed_dim=EMBED_DIM, num_heads=NUM_HEADS)

x = torch.randn(BATCH_SIZE, SEQ_LEN, EMBED_DIM)
output = mhsa(x)
print(output.shape) # torch.shape([BATCH_SIZE, SEQ_LEN, EMBED_DIM])
"""