import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class GroupedQueryAttention(nn.Module):
    def __init__(self, embed_dim : int, num_heads : int, num_kv_groups : int, dropout : float=0.1) -> None:
        super(GroupedQueryAttention, self).__init__()

        assert embed_dim % num_heads == 0, "Stream embedding dimensions should be divisible by NUM_HEADS"

        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.num_kv_groups = num_kv_groups

        self.query = nn.Linear(embed_dim, embed_dim)
        self.key = nn.Linear(embed_dim, self.head_dim * self.num_kv_groups)
        self.value = nn.Linear(embed_dim, self.head_dim * self.num_kv_groups)

        self.attention_dropout = nn.Dropout(dropout)
        
        self.out_proj = nn.Linear(embed_dim, embed_dim)
        self.proj_dropout = nn.Dropout(dropout)

    def forward(self, x : torch.Tensor, mask : torch.Tensor=None) -> torch.Tensor:
        """
        Args:
            x : Input tensor of the shape [BATCH_SIZE, SEQ_LEN, EMBED_DIM]
            mask : Mask tensor of the shape either [BATCH_SIZE, NUM_HEADS, SEQ_LEN, SEQ_LEN] or [1, NUM_HEADS, SEQ_LEN, SEQ_LEN]
        """

        BATCH_SIZE, SEQ_LEN, EMBED_DIM = x.size()

        Q = self.query(x)
        K = self.key(x)
        V = self.value(x)

        # split the Query tensor into N heads while Key and Value tensors to M groups
        Q = Q.reshape(BATCH_SIZE, SEQ_LEN, self.num_heads, self.head_dim).permute(0, 2, 1, 3)
        K = K.reshape(BATCH_SIZE, SEQ_LEN, self.num_kv_groups, self.head_dim).permute(0, 2, 1, 3)
        V = V.reshape(BATCH_SIZE, SEQ_LEN, self.num_kv_groups, self.head_dim).permute(0, 2, 1, 3)


        # repeat the tensor elements to support matrix multiplication
        heads_per_group = self.num_heads // self.num_kv_groups

        K = torch.repeat_interleave(K, heads_per_group, dim=1)
        V = torch.repeat_interleave(V, heads_per_group, dim=1)

        attention_scores = torch.matmul(Q, K.transpose(2, 3)) / math.sqrt(self.head_dim)

        if mask is not None:
            attention_scores = attention_scores.masked_fill(mask == 0, float('-inf'))

        attention_weights = F.softmax(attention_scores, dim=-1)

        attention_weights = self.attention_dropout(attention_weights)

        output = torch.matmul(attention_weights, V)

        output = output.transpose(1, 2).contiguous().view(BATCH_SIZE, SEQ_LEN, EMBED_DIM)

        out_proj = self.out_proj(output)

        out = self.proj_dropout(out_proj)

        return out



"""
BATCH_SIZE = 12
SEQ_LEN = 100
EMBED_DIM = 768
NUM_HEADS = 32
NUM_KV_GROUPS = 4

x = torch.randn(BATCH_SIZE, SEQ_LEN, EMBED_DIM)

gqa = GroupedQueryAttention(embed_dim=EMBED_DIM, num_heads=NUM_HEADS, num_kv_groups=NUM_KV_GROUPS)
print(gqa(x).shape) # [BATCH_SIZE, SEQ_LEN, EMBED_DIM]
"""