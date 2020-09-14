import pdb
import torch


def bilinear_to_linear(A):
    r"""Compute the matrix C that linearizes a bilinear function
    :math:`f(x, y) = Diag(x)Ay` for some matrix :math:`A`.

    Any bilinear function :math:`f` in two variables of size :math:`n` and
    :math:`m`, respectively, has a corresponding linear transformation in one
    variable of size :math:`nxm`, such that :math:`f(x, y) = C (x \otimes y)`

    :param A: matrix defining the bilinear form
    :type A: :class:`torch.tensor`

    :returns: matrix corresponding to the linearization
    :rtype: :class:`torch.tensor`
    
    """
    rows, cols = A.shape
    return torch.diag_embed(A.t()).reshape(rows * cols, rows).t()


def norm_grad_poly(*weights):
    r"""Compute the coefficients of the norm-gradient polynomial corresponding
    to a multilayer network, given in lexicographic order.

    :param weights: any number of weight matrices, corresponding to each
        layer in ascending order
    :type weights: :class:`torch.tensor`

    :returns: coefficients of norm-gradient polynomial
    :rtype: :class:`torch.tensor`

    """
    coeffs = bilinear_to_linear(weights[0])
    
    for weight in weights[1:]:
        coeffs = weight @ coeffs
        coeffs = bilinear_to_linear(coeffs)

    return coeffs


def propagate_box_bounds(lu, weight, bias=None):
    r"""Compute upper and lower bounds for the output of an affine
    transformation :math:`Wx + b`, given upper and lower bounds for
    the variable :math:`x`.
    
    :param lu: matrix with two columns corresponding to the lower and upper
        bounds of the variables, respectively
    :type lu: :class:`torch.tensor`
    :param weight: :math:`W` matrix of the affine transformation
    :type weight: :class:`torch.tensor`
    :param bias: translation vector :math:`b` of the affine transformation,
        defaults to None
    :type bias: :class:`torch.tensor`, optional

    :returns: upper and lower bounds for the output of the affine transformation,
        in the same format as the input parameter ``lu``.
    :rtype: :class:`torch.tensor`
    
    """
    ul = torch.flip(lu, (1,))
    w_positive = torch.clamp(weight, min=0.0)
    w_negative = torch.clamp(weight, max=0.0)
    output_lu = torch.flip(w_positive @ ul + w_negative @ lu, (1,))

    if bias is not None:
        output_lu += bias

    return lu_new


def grad_poly_bounds(
            sigma, weights, biases=None, lu=None, d_sigma_bound=None, 
            d_sigma_min=0., d_sigma_max=1.):
        r"""Compute lower and upper bounds for the variables of the
        norm-gradient polynomial, given optional lower and upper bounds
        for the input of the network.

        :param sigma: activation function
        :type sigma: callable
        :param weights: weight matrices of the network, from first to last layer
        :type weights: list
        :param biases: bias vectors of the network, from first to last layer,
            defaults to None
        :type biases: list, optional
        :param lu: lower and upper bounds for the input, given as columns of a matrix,
            defaults to None
        :type lu: :class:`torch.tensor`, optional
        :param d_sigma_bound: function returning lower and upper bounds for the
            derivative of the activation function, given lower and upper bounds
            of its input, as columns of a matrix, defaults to None
        :type d_sigma_bound: callable, optional
        :param d_sigma_min: global upper bound of the derivative of the activation,
            defaults to 0.0
        :type d_sigma_min: float, optional
        :param d_sigma_max: global lower bound of the derivative of the activation,
            defaults to 1.0
        :type d_sigma_max: float, optional

        :returns: lower and upper bounds for all variables of the norm-gradient
            polynomial, as the columns of a matrix
        :rtype: :class:`torch.tensor`

        """
        input_dim = weights[0].shape[1]
        bounds = [torch.tensor([[-1., 1.]]).expand(input_dim, 2)]

        if not lu:
            raise NotImplementedError

        if biases is None:
            biases = [None] * len(weights)
        
        for weight, bias in zip(weights, biases):
            lu = propagate_box_bounds(lu, weight, bias)
            bounds.append(d_sigma(lu))
            lu = sigma(lu)

        return torch.cat(bounds, dim=0)



def krivine_certificate(bounds, degree):
    """Generate a certificate based on Krivine's positivity certificate,
    that a polynomial is positive over a hypercube.

    :param bounds: upper and lower bounds for each variable
    :type bounds: `torch.tensor`
    :param degree: degree in the hierarchy
    :type degree: int

    :returns: matrix corresponding to the certificate's coefficients
    :rtype: `torch.tensor`

    """
    raise NotImplementedError

