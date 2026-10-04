import math
from abc import ABC, abstractmethod


class LineSearchABC(ABC):

    def __init__(self, func, eps: float = 0.002):
        self.func = func
        self.eps = eps

    @abstractmethod
    def calculate_lambda_and_u(self, left_point, right_point):
        ...

    @abstractmethod
    def determine_iteration(self, left_point, right_point):
        ...

    def update_interval(self, left_point: float, right_point: float):
        l, u = self.calculate_lambda_and_u(left_point, right_point)
        if self.func(l) > self.func(u):
            left_point = l
        else:
            right_point = u
        return left_point, right_point

    def run(self, left_point: float, right_point: float):
        print(f'===== {type(self).__name__} =====')
        n = self.determine_iteration(left_point, right_point)
        for i in range(n):
            print(f'itr {i + 1}: a = {left_point:.6f}, b = {right_point:.6f}')
            left_point, right_point = self.update_interval(left_point, right_point)
        lambda_star = (left_point + right_point) / 2
        print(f'final: a = {left_point:.8f}, b = {right_point:.8f}, '
              f'length = {right_point - left_point:.4e}')
        print(f'the number of total iterations: {n}')
        print(f'lambda* = {lambda_star:.8f}, g(lambda*) = {self.func(lambda_star):.8f}\n')
        return lambda_star, (left_point, right_point)


class DichotomousLineSearch(LineSearchABC):

    def __init__(self, func, eps: float = 0.002, delta: float = None):
        super().__init__(func, eps)
        self.delta = eps / 10 if delta is None else delta

    def calculate_lambda_and_u(self, left_point, right_point):
        mid = (left_point + right_point) / 2
        return mid - self.delta, mid + self.delta

    def determine_iteration(self, left_point, right_point):
        L0 = right_point - left_point
        if L0 <= self.eps:
            return 0
        return math.ceil(math.log2((L0 - 2 * self.delta) / (self.eps - 2 * self.delta)))


class GoldenLineSearch(LineSearchABC):

    RATIO = (math.sqrt(5) - 1) / 2

    def determine_iteration(self, left_point, right_point):
        L0 = right_point - left_point
        if L0 <= self.eps:
            return 0
        return math.ceil(math.log(self.eps / L0) / math.log(self.RATIO))

    def calculate_lambda_and_u(self, left_point, right_point):
        length = right_point - left_point
        l = left_point + (1 - self.RATIO) * length
        u = left_point + self.RATIO * length
        return l, u


class FibonacciLineSearch(LineSearchABC):

    def __init__(self, func, eps: float = 0.002, delta: float = None):
        super().__init__(func, eps)
        self.delta = eps / 10 if delta is None else delta
        self.FN_1 = self.FN_2 = self.FN_3 = 0

    def calculate_lambda_and_u(self, left_point, right_point):
        length = right_point - left_point
        l = left_point + self.FN_1 / self.FN_3 * length
        u = left_point + self.FN_2 / self.FN_3 * length
        if self.FN_3 == 2:
            u = l + self.delta

        self.FN_1, self.FN_2, self.FN_3 = self.FN_2 - self.FN_1, self.FN_1, self.FN_2
        return l, u

    def determine_iteration(self, left_point, right_point):
        L0 = right_point - left_point
        if L0 <= self.eps:
            return 0
        target = L0 / self.eps
        fib = [1, 1, 2]
        while fib[-1] < target:
            fib.append(fib[-1] + fib[-2])
        self.FN_1, self.FN_2, self.FN_3 = fib[-3], fib[-2], fib[-1]
        return len(fib) - 2

def ProblemB():
    f = lambda lam: (0 + 4.4 * lam - 2) ** 4 + ((0 + 4 * lam) + 2 * (3 - 2.4 * lam)) ** 2
    GoldenLineSearch(f, eps=EPS).run(0, 100)


if __name__ == '__main__':

    EPS = 1e-6
    f = lambda lam: (0 + 4.4 * lam - 2) ** 4 + ((0 + 4.4 * lam) - 2 * (3 - 2.4 * lam)) ** 2
    GoldenLineSearch(f, eps=EPS).run(0, 100)