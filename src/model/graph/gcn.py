import torch
import torch.nn as nn
import torch.nn.functional as F


class GraphConvolution(nn.Module):
    """
    Single GCN layer, based on https://arxiv.org/abs/1609.02907
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        bias: bool = True
    ):
        super(GraphConvolution, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.lin = nn.Linear(in_features, out_features, bias=bias)

    def forward(self, x, adj):
        x = self.lin(x)
        x = torch.matmul(adj, x)
        return x


class GCN(nn.Module):
    """
    Multiple GCN layers, based on https://arxiv.org/abs/1609.02907
    """

    def __init__(
        self,
        num_layers: int,
        in_features: int,
        hidden_features: int,
        out_features: int,
        dropout: float = 0.5,
        bias: bool = True
    ):
        super(GCN, self).__init__()

        self.convs = nn.ModuleList()
        self.convs.append(
            GraphConvolution(in_features, hidden_features, bias=bias)
        )
        for _ in range(num_layers - 2):
            self.convs.append(
                GraphConvolution(hidden_features, hidden_features, bias=bias)
            )
        self.convs.append(
            GraphConvolution(hidden_features, out_features, bias=bias)
        )

        self.dropout = dropout

    def forward(self, x, adj):
        for conv in self.convs[:-1]:
            x = conv(x, adj)
            x = F.relu(x)
            x = F.dropout(x, self.dropout, training=self.training)
        x = self.convs[-1](x, adj)
        return x
