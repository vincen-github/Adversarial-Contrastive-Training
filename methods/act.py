from torch.nn.functional import normalize
from .whitening import Whitening2d
from .base import BaseMethod
from torch import empty_like, eye, norm, cat, randperm, trace

def frobenius_inner_product(A, B):
    return trace(A.T @ B)

def act(x0, x1, lambda_param=1e-3):
    x0 = normalize(x0)
    x1 = normalize(x1)

    N = x0.size(0)
    D = x0.size(1)

    x0_norm = (x0 - x0.mean(0)) / x0.std(0)
    x1_norm = (x1 - x1.mean(0)) / x1.std(0)

    c = x0_norm.T @ x1_norm / N # DxD
    c_diff = c - eye(D, device=c.device)
    G = c_diff.detach()
    return norm(x0 - x1, p=2, dim=1).pow(2).mean() + lambda_param * frobenius_inner_product(c_diff, G)

class ACT(BaseMethod):
    """ implements our ssl loss"""
    def __init__(self, cfg):
        super().__init__(cfg)
        self.loss_f = act

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

class WhiteningACT(BaseMethod):
    """ implements whitening ACT """

    def __init__(self, cfg):
        """ init whitening transform """
        super().__init__(cfg)
        self.whitening = Whitening2d(cfg.emb, eps=cfg.w_eps, track_running_stats=False)
        self.loss_f = act
        self.w_iter = cfg.w_iter
        self.w_size = cfg.bs if cfg.w_size is None else cfg.w_size

    def forward(self, samples):
        bs = len(samples[0])
        h = [self.model(x.cuda(non_blocking=True)) for x in samples]
        h = self.head(cat(h))
        loss = 0
        for _ in range(self.w_iter):
            z = empty_like(h)
            perm = randperm(bs).view(-1, self.w_size)
            for idx in perm:
                for i in range(len(samples)):
                    z[idx + i * bs] = self.whitening(h[idx + i * bs])
            for i in range(len(samples) - 1):
                for j in range(i + 1, len(samples)):
                    x0 = z[i * bs: (i + 1) * bs]
                    x1 = z[j * bs: (j + 1) * bs]
                    loss += self.loss_f(x0, x1)
        loss /= self.w_iter * self.num_pairs
        return loss

