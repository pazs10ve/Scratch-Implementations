import torch
import torch.nn as nn


class InstanceNorm(nn.Module):
    def __init__(self, num_channels : int, eps : float=1e-5) -> None:
        super(InstanceNorm, self).__init__()

        self.eps = eps

        self.gamma = nn.Parameter(torch.ones(num_channels))
        self.beta = nn.Parameter(torch.zeros(num_channels))



    def forward(self, x : torch.Tensor) -> torch.Tensor:

        mean = x.mean(dim=(2, 3), keepdim=True)
        variance = x.var(dim=(2, 3), keepdim=True)


        x = (x - mean) / (variance + self.eps).sqrt()

        return self.gamma.view(1, -1, 1, 1) * x + self.beta.view(1, -1, 1, 1)
    


"""
batch_size = 10
num_channels = 3
img_height = 512
img_width = 512

x = torch.randn(batch_size, num_channels, img_height, img_width)
inorm = InstanceNorm(num_channels)
print(inorm(x).shape) # torch.Size([10, 3, 512, 512])
"""
