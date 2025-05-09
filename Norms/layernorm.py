import torch
import torch.nn as nn
import torch.nn.functional as f


class LayerNorm(nn.Module):
    def __init__(self, embed_dim : int, eps : float = 1e-5) -> None:
        super(LayerNorm, self).__init__()

        # marginal term to avoid divisionn by zero, if variance tends towards zero
        self.eps = eps

        # learnable parameters
        self.gamma = nn.Parameter(torch.ones(embed_dim))
        self.beta = nn.Parameter(torch.zeros(embed_dim))

    def forward(self, x : torch.Tensor) -> torch.Tensor:

        # calculate mean and variance along the embeddings dimensions
        mean = x.mean(dim=-1, keepdim=True)
        variance = x.var(dim=-1, keepdim=True)

        x = (x-mean) / (variance + self.eps).sqrt()

        return self.gamma * x + self.beta



"""
batch_size = 12
seq_len = 68
embed_dim = 768

x = torch.randn(batch_size, seq_len, embed_dim)
ln = LayerNorm(embed_dim=embed_dim)
print(ln(x).shape) # torch.Size([12, 68, 768])
"""