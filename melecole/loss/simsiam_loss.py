from torch import Tensor, mean
from torch.nn.functional import cosine_similarity
from torch.nn import Module


class SimSiamLoss(Module):
    def __init__(self, dim=1, eps=1e-8):
        super(SimSiamLoss, self).__init__()
        self.eps = eps
        self.dim = dim

    def forward(self, z1: Tensor, z2: Tensor, p1: Tensor, p2: Tensor) -> Tensor:
        # detach operation stops gradients as found to be mandatory in
        # "Exploring Simple Siamese Representation Learning" (CVPR 2021)
        # https://arxiv.org/abs/2011.10566
        z1 = z1.detach()
        z2 = z2.detach()
        return - mean(cosine_similarity(p1, z2, dim=self.dim, eps=self.eps) +
                      cosine_similarity(p2, z1, dim=self.dim, eps=self.eps)) / 2
