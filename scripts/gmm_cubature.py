import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
plt.rc('text', usetex=True)
plt.rc('text.latex', preamble=r'\usepackage{amsmath}')
from sklearn.neighbors import BallTree
from gmm import MixtureSameFamily
import torch
import torch.autograd as autograd
from scipy.linalg import hadamard
import argparse
import time

def produce_local_cubature(dimension):
    max_power = np.ceil(np.log2(dimension))
    max_dim = int(2**max_power)
    hadamard_matrix = hadamard(max_dim)
    if (dimension == max_dim):
        return np.vstack((hadamard_matrix, -hadamard_matrix))
    else:
        new_matrix = hadamard_matrix[:, :dimension]
        return np.vstack((new_matrix, -new_matrix))

def grad_log_prob(P, X):
    X = X.detach().requires_grad_(True)

    log_prob = P.log_prob(X)

    return autograd.grad(log_prob.sum(), X)[0]

def propagate_measure(measure, target_dist, local_cubature, step_size, root_two_step_size):
    n = measure.size(0)
    new_measure = root_two_step_size*torch.Tensor(local_cubature).repeat(n, 1)
    X = measure.detach().requires_grad_()
    target_dist.log_prob(X).backward(torch.ones(n))
    grads = X.grad
    next_points = (measure + grads*step_size).repeat_interleave(len(local_cubature), dim=0)
    new_measure += next_points
    return new_measure

def plot_output(P, X, d=7.0, step=0.1, save=False, i=None):
    xv, yv = torch.meshgrid([
        torch.arange(-(d+2.0), (d-2.0), step),
        torch.arange(-(d+2.0), d, step)
    ], indexing="ij")
    pos_xy = torch.cat((xv.unsqueeze(-1), yv.unsqueeze(-1)), dim=-1)
    p_xy = P.log_prob(pos_xy.to(device)).exp().unsqueeze(-1).cpu()

    df = torch.cat([pos_xy, p_xy], dim=-1).numpy()
    df = pd.DataFrame({
        'x': df[:, :, 0].ravel(),
        'y': df[:, :, 1].ravel(),
        'p': df[:, :, 2].ravel(),
    })

    mu_hat = cubature_measure.mean(0)
    S = np.cov(cubature_measure.T, bias=True)
    mu = P.mean.numpy()
    Sigma = P.variance.numpy()

    mnorm = np.linalg.norm(mu_hat - mu)
    Snorm = np.linalg.norm(np.diag(S) - Sigma)

    fig1, ax2 = plt.subplots(constrained_layout=True)
    ax2.scatter(X[:, 0], X[:, 1], c='r', s=5.)
    ax2.set_aspect("equal")
    plt.title(r'Iter %d; $\lVert \mu - \widehat{\mu} \rVert = %.3f; \lVert \Sigma - \widehat{\Sigma} \rVert = %.3f$' % (i, mnorm, Snorm))
    if save and i is not None:
        plt.savefig("../img/%d.png" % i)
        plt.close()
    else:
        plt.show()
        
        
if __name__ == "__main__":
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    parser = argparse.ArgumentParser(
        description="Run cubature-based Langevin algorithm on a 2D Gaussian mixture model")
    parser.add_argument("--stsize", type=float, default=.1, help="step size")
    parser.add_argument("--nsteps", type=int, default=1000,
                        help="number of steps")
    parser.add_argument("--npoints", type=int, default=1000,
                        help="number of points (power of two)")
    args = parser.parse_args()

    step_size = args.stsize
    dimension = 2
    number_of_steps = args.nsteps
    number_of_points = args.npoints
    no_of_recombines = number_of_points
    number_of_halving = int(np.log2(dimension)) + 1
    log_no_of_recombines = int(np.log2(no_of_recombines))
    no_of_recombines = 2**log_no_of_recombines

    if dimension != int(2**(number_of_halving - 1)):
        number_of_halving = int(np.log2(dimension)) + 2

    root_two_step_size = np.sqrt(2.0*step_size)
    points_per_recombine = (number_of_points * pow(2, number_of_halving))/no_of_recombines
    
    """
    Set up initial measure and target
    """
    np.random.seed(1337)
    torch.manual_seed(1337)

    mix = torch.distributions.Categorical(torch.Tensor([.2, .5, .3]))
    comp = torch.distributions.Independent(torch.distributions.Normal(3.*torch.randn(3,2), torch.Tensor([[1.5, .6], [1., 1.], [1.1, 1.6]])), 1)
    measure = MixtureSameFamily(mix, comp, validate_args=False)

    prior_mean = torch.Tensor([4.0] * dimension).to(device)
    prior_cov = torch.Tensor([1.0] * dimension).diag().to(device)
    prior = torch.distributions.MultivariateNormal(
        prior_mean, covariance_matrix=prior_cov)

    prior_samples = prior.sample(torch.Size([number_of_points])).to(device)

    cubature_measure = prior_samples.clone()

    local_cubature = produce_local_cubature(dimension)

    tic = time.time()
    for i in range(number_of_steps):
        #if i % 25 == 0:
        #    plot_output(measure, cubature_measure, d=10., step=.1, save=True, i=i)
        
        # Propagate cubature measure
        big_measure = propagate_measure(cubature_measure, measure, local_cubature, step_size, root_two_step_size)
            
        # Partition particles into subsets using a ball tree
        ball_tree = BallTree(big_measure, leaf_size=0.5*points_per_recombine)
        _, indices, nodes, _ = ball_tree.get_arrays()

        # Resample particles
        new_indices = [np.random.choice(indices[nd[0]: nd[1]]) for nd in nodes if nd[2]] 

        cubature_measure = torch.Tensor(big_measure[new_indices])
         
    toc = time.time()
    print("Time: %.1f seconds" % (toc - tic))
    #plot_output(measure, cubature_measure, 10.0, .1, save=True, i=1000)

    cubature_measure = np.array(cubature_measure)

    print("Number of particles: ", len(cubature_measure))
    print("")
    print("Cubature mean vector")
    print(cubature_measure.mean(0))
    print("")
    print("Cubature covariance matrix")
    print(np.cov(cubature_measure.T, bias=True))
    print("")
    print("Actual mean vector")
    print(np.array(measure.mean))
    print("")
    print("Actual variances")
    print(np.array(measure.variance))
    print("")
    print("Difference in means")
    print(np.linalg.norm(cubature_measure.mean(0) - np.array(measure.mean))) 