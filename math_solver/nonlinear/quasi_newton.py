import numpy as np

from unconstrained_base import UnconstrainedOptimizerABC


class QuasiNewtonDFP(UnconstrainedOptimizerABC):

    def reset_inner(self, y, grad):
        self.D = np.eye(self.n_dim)
        return -self.D @ grad

    def next_direction(self, y_old, y_new, grad_old, grad_new, d_old):
        p = (y_new - y_old).reshape(-1, 1)
        q = (grad_new - grad_old).reshape(-1, 1)
        Dq = self.D @ q
        C = (p @ p.T) / (p.T @ q).item() - (Dq @ Dq.T) / (q.T @ Dq).item()
        self.D = self.D + C
        return -self.D @ grad_new