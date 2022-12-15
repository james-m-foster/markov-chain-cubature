# This code is adapted from https://github.com/dilinwang820/Stein-Variational-Gradient-Descent/blob/master/python/bayesian_logistic_regression.py

import numpy as np
import scipy.io
from sklearn.model_selection import train_test_split
import numpy.matlib as nm
import torch
from scipy.linalg import hadamard
from sklearn.neighbors import BallTree
import time

'''
    Example of Bayesian Logistic Regression (the same setting as Gershman et al. 2012):
    The observed data D = {X, y} consist of N binary class labels, 
    y_t \in {-1,+1}, and d covariates for each datapoint, X_t \in R^d.
    The hidden variables \theta = {w, \alpha} consist of d regression coefficients w_k \in R,
    and a precision parameter \alpha \in R_+. We assume the following model:
        p(\alpha) = Gamma(\alpha; a, b)
        p(w_k | a) = N(w_k; 0, \alpha^-1)
        p(y_t = 1| x_t, w) = 1 / (1+exp(-w^T x_t))
'''
class BayesianLR:
    def __init__(self, X, Y, batchsize=100, a0=1, b0=0.01):
        self.X, self.Y = X, Y
        # TODO. Y in \in{+1, -1}
        self.batchsize = min(batchsize, X.shape[0])
        self.a0, self.b0 = a0, b0
        
        self.N = X.shape[0]
        self.permutation = np.random.permutation(self.N)
        self.iter = 0
    
        
    def dlnprob(self, theta):
        
        if self.batchsize > 0:
            batch = [ i % self.N for i in range(self.iter * self.batchsize, (self.iter + 1) * self.batchsize) ]
            ridx = self.permutation[batch]
            self.iter += 1
        else:
            ridx = np.random.permutation(self.X.shape[0])
            
        Xs = self.X[ridx, :]
        Ys = self.Y[ridx]
        
        w = theta[:, :-1]  # logistic weights
        alpha = np.exp(theta[:, -1])  # the last column is logalpha
        d = w.shape[1]
        
        wt = np.multiply((alpha / 2), np.sum(w ** 2, axis=1))
        
        coff = np.matmul(Xs, w.T)
        y_hat = 1.0 / (1.0 + np.exp(-1 * coff))
        
        dw_data = np.matmul(((nm.repmat(np.vstack(Ys), 1, theta.shape[0]) + 1) / 2.0 - y_hat).T, Xs)  # Y \in {-1,1}
        dw_prior = -np.multiply(nm.repmat(np.vstack(alpha), 1, d) , w)
        dw = dw_data * 1.0 * self.X.shape[0] / Xs.shape[0] + dw_prior  # re-scale
        
        dalpha = d / 2.0 - wt + (self.a0 - 1) - self.b0 * alpha + 1  # the last term is the jacobian term
        
        return torch.Tensor(np.hstack([dw, np.vstack(dalpha)]))  # % first order derivative 
    
    def evaluation(self, theta, X_test, y_test):
        theta = theta[:, :-1]
        M, n_test = theta.shape[0], len(y_test)

        prob = np.zeros([n_test, M])
        for t in range(M):
            coff = np.multiply(y_test, np.sum(-1 * np.multiply(nm.repmat(theta[t, :], n_test, 1), X_test), axis=1))
            prob[:, t] = np.divide(np.ones(n_test), (1 + np.exp(coff)))
        
        prob = np.mean(prob, axis=1)
        acc = np.mean(prob > 0.5)
        llh = np.mean(np.log(prob))
        return [acc, llh]


def propagate_measure(measure, gradlnprob, local_cubature, step_size, root_step_size):
    n = measure.size(0)
    new_measure = root_step_size*torch.Tensor(local_cubature).repeat(n, 1)
    gradients = gradlnprob(np.array(measure))
    
    # THεO POULA adjustment
    abs_gradient = torch.abs(gradients)

    gradients = torch.div(gradients, 1.0 + root_step_size*abs_gradient)
    gradients = gradients + torch.div(root_step_size*gradients, 1.0 + abs_gradient)

    next_points = (measure + gradients*step_size).repeat_interleave(len(local_cubature), dim=0)
    new_measure += next_points
    
    return new_measure

def produce_local_cubature(dimension):
    max_power = np.ceil(np.log2(dimension))
    max_dim = int(2**max_power)
    hadamard_matrix = hadamard(max_dim)
    if (dimension == max_dim):
        return np.sqrt(2.0)*np.vstack((hadamard_matrix, -hadamard_matrix))
    else:
        new_matrix = hadamard_matrix[:, :dimension]
        return np.sqrt(2.0)*np.vstack((new_matrix, -new_matrix))
    

if __name__ == '__main__':
    data = scipy.io.loadmat('../data/covertype.mat')
    
    X_input = data['covtype'][:, 1:]
    y_input = data['covtype'][:, 0]
    y_input[y_input == 2] = -1
    
    N = X_input.shape[0]
    X_input = np.hstack([X_input, np.ones([N, 1])])
    d = X_input.shape[1]
    dimension = d + 1
    
    # split the dataset into training and testing
    X_train, X_test, y_train, y_test = train_test_split(X_input, y_input, test_size=0.2, random_state=42)
    
    a0, b0 = 1, 0.01 #hyperparameters
    model = BayesianLR(X_train, y_train, 100, a0, b0) # batchsize = 100
    
    # initialization
    number_of_points = 1024
    cubature_measure = np.zeros([number_of_points, dimension]);
    alpha0 = np.random.gamma(a0, b0, number_of_points);

    for i in range(number_of_points):
        cubature_measure[i, :] = np.hstack([np.random.normal(0, np.sqrt(1 / alpha0[i]), d), np.log(alpha0[i])])
    
    cubature_measure = torch.Tensor(cubature_measure)
    local_cubature = produce_local_cubature(dimension)
    
    points_per_recombine = 128
    number_of_halving = 4
    number_of_steps = 1500
    step_size = 0.01
    root_step_size = np.sqrt(step_size)
    
    tic = time.time()
    
    for i in range(number_of_steps):
        # Propagate cubature measure
        big_measure = propagate_measure(cubature_measure, model.dlnprob, local_cubature, step_size, root_step_size)
        
        # Partition particles into subsets using a ball tree
        ball_tree = BallTree(big_measure, leaf_size=0.5*points_per_recombine)
        _, indices, nodes, _ = ball_tree.get_arrays()
        
        # Resample particles
        new_indices = [np.random.choice(indices[nd[0]: nd[1]]) for nd in nodes if nd[2]]    
        cubature_measure = torch.Tensor(big_measure[new_indices])
        
        if (i+1) % 100 == 0:
            print('iter ' + str(i+1))
            
    toc = time.time()   

    print("Time: %.1f seconds" % (toc - tic))

    print("Number of particles: ", len(cubature_measure))
    
    print("[accuracy, log-likelihood]")
    print(model.evaluation(cubature_measure, X_test, y_test))
    