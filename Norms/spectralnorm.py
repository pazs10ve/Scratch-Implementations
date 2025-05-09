import torch
import torch.nn as nn
import torch.nn.functional as F


class SpectralNorm(nn.Module):
    def __init__(self, num_features : int, num_iters : int, eps : float=1e-5) -> None:
        super(SpectralNorm, self).__init__()

        self.num_iters = num_iters
        self.eps = eps

        # vector to align with the largest singular value 
        self.u = torch.randn(num_features)

    def forward(self, W : torch.Tensor) -> torch.Tensor:

        # power iteration method to calculate the largest singular value
        for _ in range(self.num_iters):

            # v ← W·u / ||W·u||
            v = F.normalize(torch.matmul(W, self.u), dim=0)

            # u ← W^T·v / ||W^T·v||
            self.u = F.normalize(torch.matmul(W.permute(1, 0), v), dim=0)

        # σ(W) ≈ v·W·u
        sigma = torch.matmul(v, (torch.matmul(W, self.u)))

        # normalize with the largest singular value 
        return W / (sigma + self.eps)


"""
in_features = 128
out_features = 256
num_iters = 5

x = torch.randn(in_features, out_features) 
sn = SpectralNorm(num_features=out_features, num_iters=num_iters)
print(sn(x).shape) # torch.Size([128, 256])
"""