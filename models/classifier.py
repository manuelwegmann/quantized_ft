import torch.nn as nn


class LinearProbe(nn.Module):
    """
    Single linear layer for downstream multi-label classification.
    Used with backbone frozen (linear probe evaluation).
    Outputs raw logits; apply sigmoid + BCE outside.
    """

    def __init__(self, in_dim: int = 512, n_classes: int = 30):
        super().__init__()
        self.linear = nn.Linear(in_dim, n_classes)

    def forward(self, x):
        return self.linear(x)


class FineTuneHead(nn.Module):
    """
    2-layer MLP head for full fine-tuning (future use).
    Drop-in replacement for LinearProbe when backbone is unfrozen.
    """

    def __init__(self, in_dim: int = 512, hidden_dim: int = 256, n_classes: int = 30):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, n_classes),
        )

    def forward(self, x):
        return self.net(x)
