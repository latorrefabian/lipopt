# lipopt
Upper bounds on the Lipschitz constant of Neural Networks using Polynomial Optimization

Includes the original code used in the experimental section of the paper

Lipschitz constant estimation of Neural Networks via sparse polynomial optimization
Fabian Latorre, Paul Rolland, Volkan Cevher.
ICLR 2020

link: https://openreview.net/forum?id=rJe4_xSFDB

the original code can be found in the folder `old_code`, while at the top
level you can find a new (in progress) version with faster and more efficient
implementation.

IMPORTANT: a bug was found in the original version of the code, in the
`_compute_grad_poly` method in `old_code/polynomials.py`. The code
has been fixed and we expect changes in experimental results to be minor.
