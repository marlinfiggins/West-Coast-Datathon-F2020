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

def compute_ll(Delta, log_list):
    ll = np.matmul(Delta, log_list).sum()
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

    def simulate(self, beta, lambd, scoring, steps):
        '''
        This is used to simulate an iteration of our model.
        scoring: this is a function which allows us to score our agents at
        each time step, so that we can evaluate the utility of endorsement
        between agents.
        lambd: is a parameter denoting the relative weights of new endorsements
        and endorsement history.
        '''
        n = self.n
        Delta = np.zeros((steps + 1, n, n))  # Unweighted Updates
        Delta[0] = self.A0
        A = Delta.copy()  # Weighted history
        PHI = np.zeros((self.k_features, n, n))  # Features computed from score

        # Initializing variables
        self.prob_mat = np.zeros((steps, n, n))
        self.A = np.zeros((steps.n, n, n))
        self.compute_phi()

        for t in range(1, steps + 1):

            # Compute prob_mat using features
            p = np.tensordot(beta, PHI[t], axes=(0, 0))  # Taking R to [0. 1]

            prob_mat = np.exp(p)
            prob_mat = prob_mat / prob_mat.sum(axis=1)[:, np.newaxis]

            # Compute update
            Delta[t-1] = deterministic_step(prob_mat)

            # update state
            A[t] = lambd*A[t-1] + (1 - lambd)*Delta[t-1]

            self.prob_mat[t-1] = prob_mat
            self.A[t-1] = A[t-1]
            return Delta

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
        for t in range(self.steps - 1):
            #ll += (self.Delta[t].toarray()*np.log(self.prob_mat[t])).sum()
            #ll = np.matmul(self.Delta[t].toarray(), log_list[t]).sum()
            #ll += compute_ll(self.Delta[t].toarray(), log_list[t])
            ll += sparse.csr_matrix.dot(self.Delta[t], log_list[t]).sum()
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
                method='Nelder-Mead'
        )

        return res

    def objective(self, lambd):
        self.compute_state_from_deltas(lambd)
        print("Starting optimization over beta.")

        res = self.beta_max(b0=self.b0)
        out = res['fun']
        self.b0 = res['x']
        return out

    def optim(self, lambd0, alpha0=1, delta=10 ** (-1), tol=10 ** (-1), max_step=0.5):
        # We'll use a finite differences scheme to optimize this.

        # Write function that saves this and loads if already exists
        self.compute_phi()  # Features only need to be computed once.
        print("Computed Features.")

        self.b0 = np.zeros(self.k_features)

        # Initalizing for gradient ascent
        obj_old = np.inf
        obj = self.objective(lambd0)
        alpha = alpha0
        lambd = lambd0

        while (obj_old - obj > tol):
            obj_old = obj
            diff = (self.objective(lambd + delta) - obj) / delta
            step = np.sign(diff)*np.min((np.abs(diff), max_step/alpha))

            obj_prop = np.inf
            prop = lambd - alpha*step
            print(alpha*step)
            while obj_prop > obj:
                # Once we identify direction of increase find hyper parameter
                # small enough for obj increase
                prop = lambd - alpha*step
                obj_prop = self.objective(prop)
                alpha = alpha / 2

            lambd = prop
            obj = obj_prop
            alpha = alpha0  # Reset hyperparameter
            print(f"Current lambda: {lambd}")

        # After finding tolerant lambda, reoptimize
        out = self.objective(lambd)

        return({
            'lambda': lambd,
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
