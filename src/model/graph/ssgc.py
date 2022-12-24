import torch
import torch.nn as nn


class SSGC(nn.Module):
    """
    SSGC implementation, based on https://openreview.net/pdf?id=CYO5T-YjWZV
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        k: int = 2,
        alpha: float = 0.1,
        bias: bool = True
    ):
        super(SSGC, self).__init__()
        self.k = k
        self.alpha = alpha
        self.in_features = in_features
        self.out_features = out_features
        self.lin = nn.Linear(in_features, out_features, bias=bias)

    def forward(self, x, adj):
        feats = []
        for k in range(self.k):
            neighbors = torch.matmul(adj ** k, x)
            feats.append(neighbors)

        x = (1 - self.alpha) * torch.mean(torch.stack(feats), dim=0) + \
            self.alpha * x
        x = self.lin(x)
        return x
