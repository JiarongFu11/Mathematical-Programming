import numpy as np
from line_search import GoldenLineSearch


class GradientDescent:
    def __init__(self, f, grad_f, eps=1e-6):
        self.f = f
        self.grad_f = grad_f
        self.eps = eps

    @staticmethod
    def bracket(g, b=1.0, max_expand=60):
        g0 = g(0.0)
        for _ in range(max_expand):
            if g(b) > g0 or g(2 * b) > g(b):
                return 0.0, 2 * b
            b *= 2
        return 0.0, b

    def gd(self, x0, max_iter=100000):
        x = np.array(x0, dtype=float)
        path = [x.copy()]
        k = 0
        while k < max_iter:
            g = self.grad_f(x)
            if np.linalg.norm(g) < self.eps:
                break
            d = -g
            phi = lambda lam: self.f(x + lam * d)
            a, b = self.bracket(phi)
            lam, _ = GoldenLineSearch(phi, eps=self.eps).run(a, b)
            x = x + lam * d
            k += 1
            path.append(x.copy())
            if k <= 10 or k % 50 == 0:
                print(f'itr {k:4d}: x = ({x[0]:.8f}, {x[1]:.8f}), f = {self.f(x):.3e}, '
                      f'||grad|| = {np.linalg.norm(self.grad_f(x)):.3e}, lambda = {lam:.6f}')
        return x, k, np.array(path)