import matplotlib.pyplot as plt
plt.rc('text', usetex=True)
plt.rc('text.latex', preamble=r'\usepackage{amsmath}')

from gmm import MixtureSameFamily
import torch
import numpy as np
import time
import pandas as pd
import argparse

def plot_output(P, X, d=7.0, step=0.1, save=False, i=None):
    xv, yv = torch.meshgrid([
        torch.arange(-d, d, step),
        torch.arange(-d, d, step)
    ], indexing="ij")
    pos_xy = torch.cat((xv.unsqueeze(-1), yv.unsqueeze(-1)), dim=-1)
    p_xy = P.log_prob(pos_xy.to(device)).exp().unsqueeze(-1).cpu()

    df = torch.cat([pos_xy, p_xy], dim=-1).numpy()
    df = pd.DataFrame({
        'x': df[:, :, 0].ravel(),
        'y': df[:, :, 1].ravel(),
        'p': df[:, :, 2].ravel(),
    })

    mu_hat = points.mean(0)
    S = np.cov(points.T, bias=True)
    mu = P.mean.numpy()
    Sigma = P.variance.numpy()

    mnorm = np.linalg.norm(mu_hat - mu)
    Snorm = np.linalg.norm(np.diag(S) - Sigma)

    fig1, ax2 = plt.subplots(constrained_layout=True)

    ax2.scatter(X[:, 0], X[:, 1], c='r', s=5.)
    ax2.set_aspect("equal")
    plt.title(r'Iter %d; $\lVert \mu - \hat{\mu} \rVert = %.3f; \lVert \Sigma - S \rVert = %.3f$' % (i, mnorm, Snorm))
    if save and i is not None:
        plt.savefig("../img/%d.png" % i)
        plt.close()
    else:
        plt.show()

if __name__ == "__main__":
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    parser = argparse.ArgumentParser(
        description="Run unadjusted Langevin algorithm on a 2D Gaussian mixture")
    parser.add_argument("--stsize", type=float, default=.1, help="step size")
    parser.add_argument("--nsteps", type=int, default=1001000,
                        help="number of steps")
    parser.add_argument("--burnin", type=int, default=1000,
                        help="burn-in period")       
    args = parser.parse_args()

    step_size = args.stsize
    number_of_steps = args.nsteps
    burn_in = args.burnin
    dimension = 2
    
    root_two_step_size = np.sqrt(2.0*step_size)

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
    prior = torch.distributions.MultivariateNormal(prior_mean, covariance_matrix=prior_cov)

    tic = time.time()

    current_point = prior.sample().to(device)
    mu0 = np.zeros(dimension)
    Sigma0 = np.eye(dimension)        
    standard_normal = torch.distributions.MultivariateNormal(torch.Tensor(mu0).to(device),
                                                       covariance_matrix=torch.Tensor(Sigma0).to(device))

    points = torch.zeros(number_of_steps - burn_in, dimension)
    points[0] = current_point
    for i in range(number_of_steps):
        X = current_point.detach().requires_grad_()
        measure.log_prob(X).backward()
        w = standard_normal.sample().to(device)
        current_point = current_point + (X.grad)*step_size + root_two_step_size*w
        if (i >= burn_in):
            points[i-burn_in] = current_point

    toc = time.time()
    
    points = np.array(points)
    
    print("Time: %.1f seconds" % (toc - tic))
    
    print("Number of samples: ", len(points))
    print("")
    print("Empircal mean vector")
    print(points.mean(0))
    print("")
    print("Empircal covariance matrix")
    print(np.cov(points.T, bias=True))
    print("")
    print("Actual mean vector")
    print(np.array(measure.mean))
    print("")
    print("Actual variances")
    print(np.array(measure.variance))
    print("")
    print("Difference in means")
    print(np.linalg.norm(points.mean(0) - np.array(measure.mean)))
    
    #plot_output(measure, points, 15., .1, save=True, i=1000)