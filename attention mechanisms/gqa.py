import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class GroupedQueryAttention(nn.Module):
    def __init__(self, *args, **kwargs) -> None:
        super(GroupedQueryAttention, self).__init__()

        