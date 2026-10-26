from abc import ABC, abstractmethod

import numpy as np

try:
    from .line_search import GoldenLineSearch
except ImportError:
    from line_search import GoldenLineSearch


class UnconstrainedOptimizerABC(ABC):

    def __init__(self, func, gd_func, n_dim, eps=1e-6, ls_eps=1e-9, max_iter=10_000):
        self.func = func
        self.gd_func = gd_func
        self.n_dim = n_dim
        self.eps = eps
        self.ls_eps = ls_eps
        self.max_iter = max_iter

    @abstractmethod
    def reset_inner(self, y, grad):
        """每个外层迭代开始时调用，返回第一个方向 d_1。"""

    @abstractmethod
    def next_direction(self, y_old, y_new, grad_old, grad_new, d_old):
        """已知 y_i, y_{i+1} 及其梯度，返回 d_{i+1}并更新内部状态。"""

    @staticmethod
    def norm_gd(gd):
        return float(np.linalg.norm(gd))

    def line_search(self, y, d):
        phi = lambda lam: self.func(y + lam * d)

        b = 1.0
        while phi(2 * b) < phi(b):
            b *= 2
        b *= 2

        ls = GoldenLineSearch(phi, eps=self.ls_eps)
        left, right = 0.0, b
        for _ in range(ls.determine_iteration(left, right)):
            left, right = ls.update_interval(left, right)
        return (left + right) / 2

    def inner_iteration(self, x):
        y = np.asarray(x, dtype=float)
        grad = self.gd_func(y)
        d = self.reset_inner(y, grad)
        path = []
        for i in range(self.n_dim):
            lam = self.line_search(y, d)
            y_new = y + lam * d
            grad_new = self.gd_func(y_new)
            path.append(y_new)
            print(f'    i = {i + 1}: lambda = {lam:.6e}, y = {y_new}, '
                  f'||grad|| = {self.norm_gd(grad_new):.4e}')
            if self.norm_gd(grad_new) < self.eps or i == self.n_dim - 1:
                y = y_new
                break
            d = self.next_direction(y, y_new, grad, grad_new, d)
            y, grad = y_new, grad_new
        return y, path

    def run(self, start_point):
        x = np.asarray(start_point, dtype=float)
        path = [x.copy()]
        k = 0
        grad_norm = self.norm_gd(self.gd_func(x))
        print(f'k = {k}: x = {x}, f = {self.func(x):.6e}, ||grad|| = {grad_norm:.4e}')
        while grad_norm >= self.eps and k < self.max_iter:
            x, inner_path = self.inner_iteration(x)
            path.extend(inner_path)
            k += 1
            grad_norm = self.norm_gd(self.gd_func(x))
            print(f'k = {k}: x = {x}, f = {self.func(x):.6e}, ||grad|| = {grad_norm:.4e}')
        print(f'optimal x* = {x}, f(x*) = {self.func(x):.6e}, outer iterations = {k}\n')
        return x, k, np.array(path)
