import torch.nn as nn


def _norm(norm: str, dim: int, affine: bool = True) -> nn.Module:
    if norm == 'ln':
        return nn.LayerNorm(dim, elementwise_affine=affine)
    return nn.BatchNorm1d(dim, affine=affine)


class Projector(nn.Module):
    """
    3-layer MLP projector as in SimSiam.
    Architecture: Linear-Norm-ReLU → Linear-Norm-ReLU → Linear-Norm(no affine)

    norm='bn'  BatchNorm1d (original SimSiam; requires batch_size >> 1)
    norm='ln'  LayerNorm   (per-sample; batch-size agnostic)
    """

    def __init__(self, in_dim: int = 512, hidden_dim: int = 2048, out_dim: int = 2048,
                 norm: str = 'bn'):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim, bias=False),
            _norm(norm, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, hidden_dim, bias=False),
            _norm(norm, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, out_dim, bias=False),
            _norm(norm, out_dim, affine=False),
        )

    def forward(self, x):
        return self.net(x)


class Predictor(nn.Module):
    """
    2-layer bottleneck MLP predictor as in SimSiam.
    Architecture: Linear-Norm-ReLU → Linear
    Always runs in full precision — never quantized in SSQL.

    norm='bn'  BatchNorm1d (original SimSiam; requires batch_size >> 1)
    norm='ln'  LayerNorm   (per-sample; batch-size agnostic)
    """

    def __init__(self, in_dim: int = 2048, hidden_dim: int = 512, out_dim: int = 2048,
                 norm: str = 'bn'):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim, bias=False),
            _norm(norm, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, out_dim),
        )

    def forward(self, x):
        return self.net(x)
