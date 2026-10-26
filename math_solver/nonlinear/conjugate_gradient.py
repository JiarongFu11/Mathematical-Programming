from unconstrained_base import UnconstrainedOptimizerABC

class ConjugateGradientFR(UnconstrainedOptimizerABC):

    def reset_inner(self, y, grad):
        return -grad

    def next_direction(self, y_old, y_new, grad_old, grad_new, d_old):
        alpha = (grad_new @ grad_new) / (grad_old @ grad_old)
        return -grad_new + alpha * d_old
