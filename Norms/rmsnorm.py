import torch
import torch.nn as nn


class RMSNorm(nn.Module):
    def __init__(self, embed_dim : int, eps : float=1e-5) -> None:
        super(RMSNorm, self).__init__()
        self.eps = eps

        self.gamma = nn.Parameter(torch.ones(embed_dim))


    def forward(self, x : torch.Tensor) -> torch.Tensor:

        rms = (x.pow(2).mean(dim=-1, keepdim=True) + self.eps).sqrt()

        x = x / rms

        return self.gamma * x 
    


"""
batch_size = 10
seq_len = 100
embed_dim = 768

x = torch.randn(batch_size, seq_len, embed_dim)
rmsnorm = RMSNorm(embed_dim)
print(rmsnorm(x).shape) # torch.Size([10, 100, 768])
"""