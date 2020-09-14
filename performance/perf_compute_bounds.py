import torch

from utils import timing

from lipopt.utils import norm_grad_poly
from old_code.polynomials import FullyConnected

@timing
def lipopt_norm_grad_poly(*weights):
    return norm_grad_poly(*weights)

@timing
def old_norm_grad_poly(*weights):
    fc = FullyConnected(weights)
    return fc.fixed_grad_poly


def main(depth, width):
    weights = [torch.randn(width, width) for _ in range(depth - 1)]
    weights.append(torch.randn(1, width))
    lipopt_norm_grad_poly(*weights)
    old_norm_grad_poly(*weights)


if __name__ == '__main__':
    main(3, 100)

