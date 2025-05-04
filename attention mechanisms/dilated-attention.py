import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class DilatedAttention(nn.Module):
    def __init__(self, embed_dim: int, num_heads: int, dilation_factor: int, dropout: float = 0.1) -> None:
        super(DilatedAttention, self).__init__()

        assert embed_dim % num_heads == 0, "Embedding dimensions should be divisible by num_heads"

        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.dilation_factor = dilation_factor

        self.query = nn.Linear(embed_dim, embed_dim)
        self.key = nn.Linear(embed_dim, embed_dim)
        self.value = nn.Linear(embed_dim, embed_dim)

        self.attention_dropout = nn.Dropout(dropout)
        self.out_proj = nn.Linear(embed_dim, embed_dim)
        self.proj_dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, mask=None) -> torch.Tensor:
        batch_size, seq_len, _ = x.size()
        
        # Project queries, keys, values
        query = self.query(x)
        key = self.key(x)
        value = self.value(x)
        
        # Reshape for multi-head attention
        query = query.view(batch_size, seq_len, self.num_heads, self.head_dim).permute(0, 2, 1, 3)
        key = key.view(batch_size, seq_len, self.num_heads, self.head_dim).permute(0, 2, 1, 3)
        value = value.view(batch_size, seq_len, self.num_heads, self.head_dim).permute(0, 2, 1, 3)
        
        # Initialize output tensor
        attention_output = torch.zeros(batch_size, seq_len, self.embed_dim, device=x.device)
        
        # Apply dilated attention for each position
        for i in range(seq_len):
            # Calculate dilated indices for each position
            indices = torch.arange(
                max(0, i - self.dilation_factor * (seq_len // self.dilation_factor)),
                min(seq_len, i + self.dilation_factor * (seq_len // self.dilation_factor) + 1),
                self.dilation_factor,
                device=x.device
            )
            
            if len(indices) == 0:
                continue
                
            # Get query for current position and keys/values for dilated positions
            q_i = query[:, :, i, :].unsqueeze(2)  # [batch_size, num_heads, 1, head_dim]
            k_d = key[:, :, indices, :]           # [batch_size, num_heads, num_indices, head_dim]
            v_d = value[:, :, indices, :]         # [batch_size, num_heads, num_indices, head_dim]
            
            # Compute attention scores
            attention_score = torch.matmul(q_i, k_d.transpose(-1, -2)) / math.sqrt(self.head_dim)
            # [batch_size, num_heads, 1, num_indices]
            
            # Apply mask if provided
            if mask is not None:
                dilated_mask = mask[:, i:i+1, indices].unsqueeze(1)
                attention_score = attention_score.masked_fill(dilated_mask == 0, float('-inf'))
            
            # Apply softmax to get attention weights
            attention_weights = F.softmax(attention_score, dim=-1)
            attention_weights = self.attention_dropout(attention_weights)
            
            # Apply attention weights to values
            context = torch.matmul(attention_weights, v_d)  # [batch_size, num_heads, 1, head_dim]
            
            # Reshape and store the result
            context = context.transpose(1, 2).reshape(batch_size, 1, self.embed_dim)
            attention_output[:, i:i+1, :] = context
        
        # Final projection
        output = self.out_proj(attention_output)
        output = self.proj_dropout(output)
        
        return output




BATCH_SIZE = 4
SEQ_LEN = 10
EMBED_DIM = 768
NUM_HEADS = 12
DILATION_FACTOR = 2

da = DilatedAttention(embed_dim=EMBED_DIM, num_heads=NUM_HEADS, dilation_factor=DILATION_FACTOR)

x = torch.randn(BATCH_SIZE, SEQ_LEN, EMBED_DIM)
print(da(x).shape)