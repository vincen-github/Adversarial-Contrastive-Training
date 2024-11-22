from torch.nn.functional import normalize
from .whitening import Whitening2d
from .base import BaseMethod
from torch import mm, eye, cat, norm

def haochen22(x0, x1, lambda_param=1):
#    x0 = normalize(x0)
#    x1 = normalize(x1)

    N = x0.size(0)
    D = x0.size(1)

#    x0_norm = (x0 - x0.mean(0)) / x0.std(0)
#    x1_norm = (x1 - x1.mean(0)) / x1.std(0)
    
    c0 = x0.T @ x0 / N
    c1 = x1.T @ x1 / N # DxD
    c_diff = (1 / 2 * c0 + 1 / 2 * c1 - eye(D, device=c0.device)).pow(2)
    return norm(x0 - x1, p=2, dim=1).pow(2).mean() + lambda_param * c_diff.sum()


class Haochen22(BaseMethod):
    """ implements our ssl loss"""
    def __init__(self, cfg):
        super().__init__(cfg)
        self.loss_f = haochen22

    def forward(self, samples):
        bs = len(samples[0])
        h = [self.model(x.cuda(non_blocking=True)) for x in samples]
        h = self.head(cat(h))
        loss = 0
        for i in range(len(samples) - 1):
            for j in range(i + 1, len(samples)):
                x0 = h[i * bs: (i + 1) * bs]
                x1 = h[j * bs: (j + 1) * bs]
                loss += self.loss_f(x0, x1)
        loss /= self.num_pairs
        return loss