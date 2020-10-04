# hierarchy_model.py
# Description: Simulates hierarchy model based on Kawakatsu et al.
# based on https://github.com/PhilChodrow/prestige_reinforcement/blob/master/py/model.py
# with change to class structure and optimization method, underlying structure is mostly unchanged.

# Date: September 30, 2020
# Author: Marlin Figgins

# To do:
# Implement scoring based on statistical model
# -We generate scores based on several covariates which are entered as a matrix for each actor
# -We then use features in the model phi x is x is good example
# Finish implementing set_featurse and compute features
# Implement get_ methods for visualization
# Simulate heirarchies based on this model for Actor netowrks.

import numpy as np
from numba import jit

# Inference
from scipy.special import gammaln
from scipy.optimize import minimize
import scipy.sparse as sparse
import sklearn.metrics.pairwise
import warnings

@jit(nopython=True)
def deterministic_step(prob_mat, endorse_per_agent=1):
    '''
    Computes the matrix Delta(t) given a matrix of probabilities
    for endorsements between individuals i and j.

    endorse_per_agent gives the total number of endorsesment that each individual is allowed to give per time step.
    '''
    n = prob_mat.shape[0]  # self.n?
    Delta = prob_mat*endorse_per_agent / n
    return Delta

@jit(nopython=True)
def compute_ll(Delta, log_list):
    ll = 0
    for i in range(Delta.shape[0]):
        for j in range(Delta.shape[0]):
            ll += Delta[i,j]*log_list[i,j]
    return ll

class hierarchy_model:
    def __init__(self, Delta=None, A0=None, cov=None, feature_list=None):
        self.input_data(Delta, A0)
        self.input_covariates(cov)
        self.set_features(feature_list)

    def input_data(self, Delta, A0):
        self.Delta = Delta
        if A0 is None:
            self.A0 = Delta[0]
        else:
            self.A0 = A0
        if self.Delta is not None:
            self.steps = Delta.shape[0]
            self.n = Delta[0].shape[0]

    def input_covariates(self, cov=None):
        self.cov = cov
        if self.cov is not None:
            self.k_covs = cov[0].shape[1]

    def set_features(self, feature_list=None):
        '''
        feature_list is a list of functions with
        Inputs - cov: n x k_cov matrices
        Outputs - u: n x n matrices
        '''
        # For simplicity, do each pairwise differences
        if feature_list is None:
            feature_list = self.generate_features()

        self.phi = feature_list
        self.k_features = len(feature_list)

    def generate_features(self):
        '''
        Generate feature list of f(x) = x for k_covs
        '''
        def pairwise_vector_diff(v):
            v = v.reshape(-1, 1)
            #return sklearn.metrics.pairwise.pairwise_distances(v.reshape(-1,1), v.reshape(-1,1))
            return sparse.csr_matrix(sklearn.metrics.pairwise.pairwise_distances(v, v))

        feature_list = [pairwise_vector_diff]*self.k_covs
        return feature_list

    def simulate(self, beta, lambd, steps):
        '''
        This is used to simulate an iteration of our model.
        scoring: this is a function which allows us to score our agents at
        each time step, so that we can evaluate the utility of endorsement
        between agents.
        lambd: is a parameter denoting the relative weights of new endorsements
        and endorsement history.
        '''
        self.steps = steps
        self.A = self.compute_state_from_deltas(lambd)
        self.compute_phi()
        self.compute_prob_mat(beta)

        return self.prob_mat

    # Inference process functions
    def compute_state_from_deltas(self, lambd, A0=None):
        if A0 is None:
            A0 = self.Delta[0]

        A = np.zeros_like(self.Delta)
        A[0] = A0

        for t in range(1, self.steps):
            A[t] = lambd*A[t-1] + (1 - lambd)*self.Delta[t-1]
        return A

    def compute_phi(self):
        '''
        Compute relevent features based on covariates at that time.
        Each phi must take n_agents by k_cov to feature vector of interest.
        phi[1] -> f(cov) = (cov - cov.T)
        '''
        #self.PHI = np.zeros((self.steps, self.k_features, self.n, self.n))
        self.PHI = []
        for t in range(self.steps):
            placeholder = []
            for j in range(self.k_features):
                #self.PHI[t][j] = self.phi[j](self.cov[t][:, j])
                placeholder.append(self.phi[j](self.cov[t][:, j]))
            self.PHI.append(placeholder)
            print(f"Features at time {t} computed.")

    def compute_trajectory(self, lambd):
        '''
        Compute the trajectory A associated with a given lambda value
        '''
        self.A = self.state_from_deltas(lambd, self.A0)

    def compute_prob_mat(self, beta):
        '''
        Compute probability matrix corresponding to the weighted features.
        '''

        log_list = []
        # Compute rates from beta
        for t in range(self.steps):
            self.prob_mat = [0]*self.steps
            p = [0]*self.k_features
            #p = sum([beta[j]*self.PHI[t][j] for j in range(self.k_features)])
            for j in range(self.k_features):
                p[j] = beta[j]*self.PHI[t][j]

            self.prob_mat[t] = np.exp(sum(p).toarray())
            self.prob_mat[t] = self.prob_mat[t]/ self.prob_mat[t].sum(axis =1)[:, np.newaxis]
            log_list.append(np.log(self.prob_mat[t]))

        return log_list

    # Algorithm of optimization

    def likelihood(self, beta):
        '''
        Calculate likelihood for given beta vector
        '''
        warnings.filterwarnings("ignore", category=RuntimeWarning)
        log_list = self.compute_prob_mat(beta)
        ll = 0
        for t in range(1, self.steps - 1):
            #ll += (self.Delta[t].toarray()*np.log(self.prob_mat[t])).sum()
            #ll = np.matmul(self.Delta[t].toarray(), log_list[t]).sum()
            #ll += self.Delta[t].multiply(log_list[t]).sum()
            ll += compute_ll(self.Delta[t].toarray(), log_list[t])
        print(ll)
        return -ll


    def beta_max(self, b0=None):
        '''
        Estimate beta vector.
        '''

        if b0 is None:
            b0 = np.zeros(self.k_features) + 0.01

        res = minimize(
                fun=lambda b: self.likelihood(b),
                x0=b0,
                method='Nelder-Mead',
                tol=1e-2
        )
        print(res.message)
        return res

    def objective(self, lambd, tol):
        self.compute_state_from_deltas(lambd)
        print("Starting optimization over beta.")

        res = self.beta_max(b0=self.b0)
        out = res['fun']
        self.b0 = res['x']
        return out

    def optim(self, lambd0, alpha0=1, delta=10 ** (-1), tol=10 ** (-2), max_step=0.5):
        # We'll use a finite differences scheme to optimize this.

        # Write function that saves this and loads if already exists
        self.compute_phi()  # Features only need to be computed once.
        print("Computed Features.")

        self.b0 = np.zeros(self.k_features)

        # Initalizing for gradient ascent
        out = self.objective(lambd0, tol)
        # alpha = alpha0
        # lambd = lambd0

        # while (obj_old - obj > tol):

        #    obj_old = obj
        #   alpha = alpha0
        #  diff = (self.objective(lambd + delta) - obj) / delta
        # diff= np.sign(diff)*np.min((np.abs(diff), max_step/alpha))
        # print(diff)
        # obj_prop = np.inf
        # prop = lambd - alpha*diff

        # while obj_prop > obj:
        # Once we identify direction of increase find hyper parameter
        # small enough for obj increase
        #   prop = lambd - alpha*diff
        #  obj_prop = self.objective(prop)
        # alpha = alpha / 2

        # lambd = prop
        # obj = obj_prop

        # print(f"Current lambda: {lambd}. LL: {obj}")

        # After finding tolerant lambda, reoptimize
        # out = self.objective(lambd)

        return({
            'beta': self.b0.tolist(),
            "LL": out
        })

    # Inspection methods
    def get_Delta(self):
        pass

    def get_scores(self):
        pass

    def get_states(self):
        pass
