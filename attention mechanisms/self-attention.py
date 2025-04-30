import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class VanilaSelfAttention(nn.Module):
    def __init__(self, embed_dim : int, dropout : float=0.1) -> None:
        super(VanilaSelfAttention, self).__init__()

        self.query = nn.Linear(embed_dim, embed_dim, bias=False)
        self.key = nn.Linear(embed_dim, embed_dim, bias=False)
        self.value = nn.Linear(embed_dim, embed_dim, bias=False)

        self.attention_dropout = nn.Dropout(dropout)

        self.out_proj = nn.Linear(embed_dim, embed_dim, bias=False)
        self.proj_dropout = nn.Dropout(dropout)
        

    def forward(self, x : torch.Tensor, mask : torch.Tensor=None) -> torch.Tensor:
        """
        Args:
            x: Input tensor of shape (batch_size, seq_len, embed_dim)
            mask: Optional mask tensor of shape (batch_size, 1, seq_len, seq_len) or (1, 1, seq_len, seq_len)
        Returns:
            Output tensor of shape (batch_size, seq_len, embed_dim)
        """
        batch_size, seq_len, embed_dim = x.size()


        # Compute query, key, value matrices
        Q = self.query(x)
        K = self.key(x)
        V = self.value(x)


        # Compute attention scores
        scores = torch.matmul(Q, K.transpose(1, 2)) / math.sqrt(embed_dim)

        # apply mask if provided
        if mask is not None:
            scores = scores.masked_fill(mask == 0, float('-inf'))

        # Apply softmax to get attention weights
        attention_weights = F.softmax(scores, dim=-1)

        attention_weights = self.attention_dropout(attention_weights)

        # Compute the attention output
        output = torch.matmul(attention_weights, V)

        # project the attention output to the input stream 
        output = self.out_proj(output)
        output = self.proj_dropout(output)

        return output




"""
BATCH_SIZE = 2
SEQ_LEN = 10
EMBED_DIM = 64

x = torch.randn(BATCH_SIZE, SEQ_LEN, EMBED_DIM)

attention = VanilaSelfAttention(embed_dim=EMBED_DIM, dropout=0.1)
output = attention(x)
print("Output shape:", output.shape)  # Expected shape: (BATCH_SIZE, SEQ_LEN, EMBED_DIM)
"""