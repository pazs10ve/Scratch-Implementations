import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class BlockWiseAttention(nn.Module):
    def __init__(self, embed_size: int, num_heads: int, block_size: int, dropout: float = 0.1, causal: bool = False):
        super(BlockWiseAttention, self).__init__()
        assert embed_size % num_heads == 0, "embed_size must be divisible by num_heads"

        self.num_heads = num_heads
        self.head_dim = embed_size // num_heads
        self.block_size = block_size
        self.causal = causal

        self.query = nn.Linear(embed_size, embed_size, bias=False)
        self.key = nn.Linear(embed_size, embed_size, bias=False)
        self.value = nn.Linear(embed_size, embed_size, bias=False)

        self.attention_dropout = nn.Dropout(dropout)
        self.out_proj = nn.Linear(embed_size, embed_size)
        self.proj_dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, mask: torch.Tensor = None) -> torch.Tensor:
        batch_size, seq_len, embed_size = x.size()

        # Pad sequence if seq_len is not divisible by block_size
        if seq_len % self.block_size != 0:
            pad_len = self.block_size - (seq_len % self.block_size)
            x = F.pad(x, (0, 0, 0, pad_len))  # Pad along sequence dimension

            if mask is not None:
                mask = F.pad(mask, (0, pad_len, 0, pad_len), value=0)
        else:
            pad_len = 0

        # Reshape into blocks
        num_blocks = (seq_len + pad_len) // self.block_size
        x_blocks = x.view(batch_size, num_blocks, self.block_size, embed_size)

        # Compute Q, K, V for all blocks at once
        query = self.query(x_blocks)  # (batch_size, num_blocks, block_size, embed_size)
        key = self.key(x_blocks)
        value = self.value(x_blocks)

        # Reshape for multi-head attention
        query = query.view(batch_size, num_blocks, self.block_size, self.num_heads, self.head_dim).transpose(2, 3)
        key = key.view(batch_size, num_blocks, self.block_size, self.num_heads, self.head_dim).transpose(2, 3)
        value = value.view(batch_size, num_blocks, self.block_size, self.num_heads, self.head_dim).transpose(2, 3)
        # Shape: (batch_size, num_blocks, num_heads, block_size, head_dim)

        # Compute attention scores
        attention_scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(self.head_dim)
        # Shape: (batch_size, num_blocks, num_heads, block_size, block_size)

        # Apply causal mask if needed
        if self.causal:
            causal_mask = torch.triu(torch.ones(self.block_size, self.block_size, device=x.device), diagonal=1)
            attention_scores = attention_scores.masked_fill(causal_mask == 1, float('-inf'))

        # Apply padding/key mask
        if mask is not None:
            # Assume mask is (batch_size, seq_len, seq_len) or (batch_size, seq_len)
            mask = mask.view(batch_size, num_blocks, self.block_size, -1)[:, :, :, :self.block_size]
            mask = mask.unsqueeze(2)  # (batch_size, num_blocks, 1, block_size, block_size)
            attention_scores = attention_scores.masked_fill(mask == 0, float('-inf'))

        # Softmax and dropout
        attention_weights = F.softmax(attention_scores, dim=-1)
        attention_weights = self.attention_dropout(attention_weights)

        # Compute attention output
        attention_output = torch.matmul(attention_weights, value)
        # Shape: (batch_size, num_blocks, num_heads, block_size, head_dim)

        # Reshape back to (batch_size, num_blocks, block_size, embed_size)
        attention_output = attention_output.transpose(2, 3).reshape(batch_size, num_blocks, self.block_size, embed_size)

        # not implemented: pool each block and compute attention across blocks

        # Flatten blocks
        attention_output = attention_output.view(batch_size, -1, embed_size)

        # Remove padding
        if pad_len > 0:
            attention_output = attention_output[:, :seq_len, :]

        # Final projection
        output = self.out_proj(attention_output)
        output = self.proj_dropout(output)

        return output


"""
batch_size = 4
embed_size = 768
seq_len = 100
block_size = 25
num_heads = 12

bwa = BlockWiseAttention(embed_size=embed_size, num_heads=num_heads, block_size=block_size, causal=True)
x = torch.randn(batch_size, seq_len, embed_size) # torch.size([4, 100, 768])
print(bwa(x).shape)  # output : torch.Size([4, 100, 768])
"""