import torch
import torch.nn.functional as F


def negative_cosine_similarity(p: torch.Tensor, z: torch.Tensor) -> torch.Tensor:
    """
    SimSiam loss D(p, z) = -cos_sim(p, z).
    Stop-gradient on z must be applied by the caller before passing z here.
    """
    p = F.normalize(p, dim=-1)
    z = F.normalize(z, dim=-1)
    return -(p * z).sum(dim=-1).mean()
