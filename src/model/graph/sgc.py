import torch
import torch.nn as nn


class SGC(nn.Module):
    """
    SGC implementation, based on https://arxiv.org/abs/1902.07153
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        k: int = 2,
        alpha: float = 0.1,
        bias: bool = True
    ):
        super(SGC, self).__init__()
        self.k = k
        self.alpha = alpha
        self.in_features = in_features
        self.out_features = out_features
        self.lin = nn.Linear(in_features, out_features, bias=bias)

    def forward(self, x, adj):
        output = torch.matmul(adj ** self.k, x)
        output = (1 - self.alpha) * output + self.alpha * x
        output = self.lin(output)
        return x
