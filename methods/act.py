from torch.nn.functional import normalize
from .whitening import Whitening2d
from .base import BaseMethod
from numpy import sqrt
from torch import empty_like, eye, norm, cat, randperm, trace
def frobenius_inner_product(A, B):
    return trace(A.T @ B)

def act(x0, x1, G, lambda_param=5e-2):
#    x0 = normalize(x0)
#    x1 = normalize(x1)

    N = x0.size(0)
    D = x0.size(1)
    x0 = sqrt(D) * normalize(x0)
    x1 = sqrt(D) * normalize(x1)

#    x0_norm = (x0 - x0.mean(0)) / x0.std(0)
#    x1_norm = (x1 - x1.mean(0)) / x1.std(0)

    c = x0.T @ x1 / N # DxD
    c_diff = c - eye(D, device=c.device)
    if G == None:
        G = c_diff
    return norm(x0 - x1, p=2, dim=1).pow(2).mean() + lambda_param * frobenius_inner_product(c_diff, G), c_diff.detach()

class ACT(BaseMethod):
    """ implements our ssl loss"""
    def __init__(self, cfg):
        super().__init__(cfg)
        self.cfg = cfg
        self.loss_f = act

    def forward(self, samples):
#        bs = len(samples[0])
        h = [self.model(x.cuda(non_blocking=True)) for x in samples]
        h = self.head(cat(h))
        loss = 0
        G = None
        for i in range(len(samples) - 1):
            for j in range(i + 1, len(samples)):
                x0 = h[i * self.cfg.bs: (i + 1) * self.cfg.bs]
                x1 = h[j * self.cfg.bs: (j + 1) * self.cfg.bs]
                loss_increments, G = self.loss_f(x0, x1, G)
                loss += loss_increments              
        loss /= self.num_pairs
        return loss