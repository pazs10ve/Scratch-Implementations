import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class MultiScaleWindowAttention(nn.Module):
    def __init__(self, )