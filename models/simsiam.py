import torch.nn as nn


class Projector(nn.Module):
    """
    3-layer MLP projector as in SimSiam.
    Architecture: Linear-BN-ReLU → Linear-BN-ReLU → Linear-BN(no affine)
    Both backbone and projector are quantized on the prediction side in SSQL.
    """

    def __init__(self, in_dim: int = 512, hidden_dim: int = 2048, out_dim: int = 2048):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim, bias=False),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, hidden_dim, bias=False),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, out_dim, bias=False),
            # No affine on the last BN — standard SimSiam design
            nn.BatchNorm1d(out_dim, affine=False),
        )

    def forward(self, x):
        return self.net(x)


class Predictor(nn.Module):
    """
    2-layer bottleneck MLP predictor as in SimSiam.
    Architecture: Linear-BN-ReLU → Linear
    Always runs in full precision — never quantized in SSQL.
    """

    def __init__(self, in_dim: int = 2048, hidden_dim: int = 512, out_dim: int = 2048):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim, bias=False),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, out_dim),
        )

    def forward(self, x):
        return self.net(x)
