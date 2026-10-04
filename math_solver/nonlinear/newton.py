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
            


def f(x):
    return (x[0] - 2) ** 4 + (x[0] - 2 * x[1]) ** 2

def grad_f(x):
    return np.array([4 * (x[0] - 2) ** 3 + 2 * (x[0] - 2 * x[1]),
                     -4 * (x[0] - 2 * x[1])])

def hess_f(x):
    return np.array([[12 * (x[0] - 2) ** 2 + 2, -4.0],
                     [-4.0, 8.0]])


if __name__ == '__main__':
    x_star, n_iter, path = Newton(grad_f, hess_f, f, eps=1e-6).run_with_line_search((0.0, 3.0))
    print(f'\nconverged after {n_iter} iterations, x* = {x_star}, f = {f(x_star):.3e}')

    fig, axes = plt.subplots(1, 2, figsize=(13, 5.5))
    views = [((-0.5, 3.0), (0.0, 3.3), 'Full path'),
             ((1.95, 2.02), (0.975, 1.01), 'Zoom near the minimum')]
    for ax, (xl, yl, title) in zip(axes, views):
        X1, X2 = np.meshgrid(np.linspace(*xl, 400), np.linspace(*yl, 400))
        ax.contour(X1, X2, (X1 - 2) ** 4 + (X1 - 2 * X2) ** 2,
                   levels=np.logspace(-9, 2, 40), cmap='viridis', linewidths=0.6)
        ax.plot(path[:, 0], path[:, 1], 'r.-', lw=1, ms=5, label='Newton path')
        ax.plot(*path[0], 'bs', ms=7, label='start (0, 3)')
        ax.plot(2, 1, 'k*', ms=12, label='optimum (2, 1)')
        ax.set_xlim(xl); ax.set_ylim(yl)
        ax.set_xlabel('$x_1$'); ax.set_ylabel('$x_2$')
        ax.set_title(title); ax.legend(loc='lower right')
    fig.suptitle(f"Gloden Line Search ($\\lambda_k=1$), {n_iter} iterations")
    fig.tight_layout()
    fig.savefig('newton_path.png', dpi=150)
    plt.show()