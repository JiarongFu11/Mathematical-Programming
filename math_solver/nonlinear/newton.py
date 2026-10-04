import numpy as np
import matplotlib.pyplot as plt
from line_search import GoldenLineSearch


class Newton():
    def __init__(self, F, J, f=None, eps=1e-6):
        self.eps = eps
        self.F = F
        self.J = J
        self.f = f
    
    @staticmethod
    def bracket(g, b=1.0, max_expand=60):
        g0 = g(0.0)
        for _ in range(max_expand):
            if g(b) > g0 or g(2 * b) > g(b):
                return 0.0, 2 * b
            b *= 2
        return 0.0, b
        
    def run(self, x0, max_iters=1000):
        x = np.asarray(x0, dtype=float)
        path = [x.copy()]
        for i in range(1, max_iters + 1):
            x = x - np.linalg.solve(self.J(x), self.F(x))
            path.append(x)
            print(f'itr {i:3d}: x = ({x[0]:.8f}, {x[1]:.8f}), '
                  f'||grad|| = {np.linalg.norm(self.F(x)):.3e}')
            if np.linalg.norm(self.F(x)) < self.eps:
                return x, i, np.array(path)
        raise RuntimeError(f'未收敛: x={x}')
    
    def run_with_line_search(self, x0, max_iters=1000):
        x = np.asarray(x0, dtype=float)
        path = [x.copy()]
        for k in range(1, max_iters + 1):
            d = -np.linalg.solve(self.J(x), self.F(x))
            phi = lambda lam: self.f(x + lam * d)
            a, b = self.bracket(phi)
            lam, _ = GoldenLineSearch(phi, eps=self.eps).run(a, b)
            x = x + lam * d
            path.append(x.copy())
            print(f'itr {k:3d}: lambda = {lam:.6f}, x = ({x[0]:.8f}, {x[1]:.8f}), '
                  f'||grad|| = {np.linalg.norm(self.F(x)):.3e}')
            if np.linalg.norm(self.F(x)) < self.eps:
                return x, k, np.array(path)
        raise RuntimeError(f'未收敛: x={x}')