import pytest
import torch

from lipopt import utils


@pytest.fixture
def W1():
    return torch.tensor([[1., 2., 3.], [-1., 0., 1.]])

@pytest.fixture
def W2():
    return torch.tensor([[1., 2.]])

@pytest.fixture
def lu_1():
    return torch.tensor([[-1., 1.], [-1., 1.], [-1., 1.]])


def test_bilinear_to_linear(W1):
    result = utils.bilinear_to_linear(W1)
    result -= torch.tensor([[1., 0., 2., 0, 3., 0.], [0., -1., 0., 0., 0., 1.]])
    assert torch.allclose(result, torch.zeros(2, 6))

def test_norm_grad_poly(W1, W2):
    result = utils.norm_grad_poly(W1, W2)
    result -= torch.tensor([[1, -2, 2, 0., 3, 2]])
    assert torch.allclose(result, torch.zeros(1, 6))
    
def test_propagate_box_bounds(lu_1, W1):
    result = utils.propagate_box_bounds(lu_1, W1)
