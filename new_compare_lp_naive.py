# import pdb
import numpy as np
import old_code as po
import torch

from old_code import utils
from timeit import default_timer as timer


# layer_configs = [(20, 20, 1), ]
width = 10
layer_configs = [(width, width, 1), ]

# layer_configs = [(2, 5, 1), ]
repeats = 1


def compare_bounds(layer_configs):
    """
    This is the default setting where we compute upper bounds
    for the Lipschitz constant over the hypercube [0, 1]^d.
    However, we might as well pass different upper and lower
    bounds corresponding to a different 'bounding box' using
    the variables lb (lower bound), and ub (upper bound).
    They correspond to vectors with bounds for each coordinate.
    This should be the only change in order to obtain bounds
    on Local lipschitz constant on sets of that form.
    """
    lb = np.repeat(0., width)  # lower bounds for domain
    ub = np.repeat(1., width)  # upper bounds for domain
    results = dict()
    for layer_config in layer_configs:
        res = []
        tight_s = []
        tight_p = []
        for _ in range(repeats):
            network = utils.fc(layer_config)
            weights, biases = utils.weights_from_pytorch(network)
            fc = po.FullyConnected(weights, biases)
            f = fc.grad_poly
            g, lb, ub = fc.new_krivine_constr(p=1, lb=lb, ub=ub)
            start = timer()
            m = po.KrivineOptimizer.new_maximize(
                    f, g, lb=lb, ub=ub, deg=len(weights),
                    start_indices=fc.start_indices,
                    layer_config=layer_config,
                    solver='gurobi', n_jobs=-1, name='')
            end = timer()
            print('time elapsed: ', end - start)
            # m = po.KrivineOptimizer.maximize(
            #        f, g, deg=len(weights),
            #        solver='gurobi', n_jobs=-1, name='')
            lp_bound = m.objVal
            print('LP BOUND: ', lp_bound)
            ubp = po.upper_bound_product(fc.weights, p=1)
            lbp = po.lower_bound_product(fc.weights, p=1)
            lbs = po.lower_bound_sampling(
                    network, layer_config[0], n=50000, p=1)
            print('LOWER BOUND PRODUCT: ', lbp)
            print('LOWER BOUND SAMPLING: ', lbs)
            return

            res.append(ubp / lp_bound)
            tight_s.append(lp_bound / lbs)
            tight_p.append(lp_bound / lbp)

        results[layer_config] = {
                'lp/sampling': sum(tight_s) / len(tight_s),
                'lp/product': sum(tight_p) / len(tight_p),
                'product/lp': sum(res) / len(res)
                }

    return results


def main():
    # np.random.seed(4)
    seed = 7
    np.random.seed(seed)
    torch.manual_seed(seed)
    results = compare_bounds(layer_configs)
    print(results)


if __name__ == '__main__':
    main()

