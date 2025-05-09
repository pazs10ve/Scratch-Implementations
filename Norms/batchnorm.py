import torch
import torch.nn as nn


class BatchNorm(nn.Module):
    def __init__(self, num_channels : int, momentum : float=0.1, eps : float=1e-5) -> None:
        super(BatchNorm, self).__init__()
        self.eps = eps
        self.momentum = momentum
        self.num_channels = num_channels

        self.gamma = nn.Parameter(torch.ones(num_channels))
        self.beta = nn.Parameter(torch.zeros(num_channels))

        self.register_buffer('running_mean', torch.zeros(num_channels))
        self.register_buffer('running_variance', torch.ones(num_channels))


    def forward(self, x : torch.Tensor) -> torch.Tensor:

        # x shape : (N, C, H, W)
        if self.training:
            mean = x.mean(dim=(0, 2, 3), keepdim=True) # shape : (1, -1, 1, 1)
            variance = x.var(dim=(0, 2, 3), keepdim=True) # shape : (1, -1, 1, 1)

            # update the running buffers
            with torch.no_grad():
                self.running_mean = (1 - self.momentum) *  self.running_mean + self.momentum * mean.squeeze()
                self.running_variance = (1 - self.momentum) * self.running_variance + self.momentum * variance.squeeze()

        # use running statistic during the inference
        else:
            mean = self.running_mean.view(1, -1, 1, 1)
            variance = self.running_variance.view(1, -1, 1, 1)

        # normalize the input vector
        x = (x - mean) / (variance + self.eps).sqrt()

        # scale and shift
        return self.gamma.view(1, -1, 1, 1) * x + self.beta.view(1, -1, 1, 1)
    

"""
batch_size = 10
num_channels = 3
img_height = 512
img_width = 512

x = torch.randn(batch_size, num_channels, img_height, img_width)
bn = BatchNorm(num_channels)
print(bn(x).shape)
"""