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
# Implement get_ methods for visualization
# Simulate heirarchies based on this model for Actor netowrks.

import numpy as np
from numba import jit

# Inference
from scipy.special import gammaln
from scipy.optimize import minimize


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


class hierarchy_model:
    def __init__(self, Delta=None, A0=None):
        self.input_data(Delta, A0)

    def input_data(self, Delta, A0):
        self.Delta = Delta
        if A0 is None:
            self.A0 = Delta[0]
        else:
            self.A0 = A0
        if self.Delta is not None:
            self.steps = Delta.shape[0]
            self.n = Delta.shape[1]

    def set_features(self, feature_list):
        pass

    def score(self, A, scoring):
        '''
        This function is used to score individuals in order to determine
        their probability of endorsing another agent.
        '''
        s = np.zeros_like(A)
        return s

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

        # initializing variables
        self.prob_mat = np.zeros((steps, n, n))
        self.A = np.zeros((steps.n, n, n))
        self.S = np.zeros((steps.n, n))

        for t in range(1, steps + 1):
            # compute scores
            s = self.score(A[t-1], scoring)
            self.S[t-1] = s

            # compute features
            # Need to take covariates to features
            # Should also loop over i and j
            for k in range(self.k_features):
                PHI[k] = self.phi[k](s)

            # compute prob_mat using scores
            p = np.tensordot(beta, PHI, axes=(0, 0))  # Taking R to [0. 1]

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

        A = np.zeros_like(self.Delta).astype(float)
        for t in range(1, self.steps + 1):
            A[t] = lambd*A[t-1] + (1 - lambd)*self.Delta[t-1]
        return A

    def compute_score(self):
        '''
        Compute scores for individual actors based on input data.
        '''

        # This may become a reference which splits covariates at relevant t.
        pass

    def compute_phi(self):
        '''
        Compute relevent features based on score input
        '''
        # phi returns feature list of interest between pairs of actors
        pass

    def compute_trajectory(self, lambd):
        '''
        Compute the trajectory A associated with a given lambda value
        '''
        self.A = self.state_from_deltas(lambd, self.A0)

    def compute_prob_mat(self, beta):
        '''
        Compute probability matrix corresponding to the scores.
        '''

        # Compute rates from beta
        p = np.tensordot(beta, self.S, axes=(0, 1))
        prob_mat = np.exp(p)  # Taking R to [0. 1]
        prob_mat = prob_mat / prob_mat.sum(axis=2)[:, :, np.newaxis]
        self.prob_mat = prob_mat

    # Algorithm of optimization
    def likelihood(self, beta):
        '''
        Calculate likelihood for given beta vector
        '''

        self.compute_prob_mat(beta)
        DeltaDiff = np.diff(self.Delta, axis=0)
        C = gammaln(DeltaDiff.sum(axis=1)+1).sum() - gammaln(DeltaDiff+1).sum()
        ll = DeltaDiff*np.log(self.prob_mat[:-1]).sum() + C
        return ll

    def beta_max(self, b0=None):
        '''
        Estimate beta vector.
        '''

        if b0 is None:
            b0 = np.zeros(self.k_features)

        res = minimize(
                fun=lambda b: -self.ll(b),
                x0=b0
        )

        return res

    def optim(self, lambd0, alpha0=10 ** (-4), delta=10 ** (-4), tol=10 ** (-3), max_step=0.2):
        # We'll use a finite differences scheme to optimize this.
        def objective(lambd):
            self.compute_state_from_deltas(lambd)
            self.compute_score()

            res = self.beta_max(b0=self.b0)
            out = res['fun']
            self.b0 = res['x']
            return out

        # Initalizing for gradient ascent
        obj_old = np.inf
        obj = objective(lambd0)
        alpha = alpha0
        lambd = lambd0

        while (obj_old - obj > tol):
            obj_old = obj
            diff = (objective(lambd + delta) - obj) / delta
            step = np.sign(diff)*min(alpha*abs(diff), max_step)

            obj_prop = np.inf
            while obj_prop > obj:
                # Once we idenitfy direction of increase find hyper parameter
                # small enough for obj increase
                prop = lambd - alpha*step
                obj_prop = objective(prop)
                alpha = alpha / 2

            obj = objective(prop)
            alpha = alpha0  # Reset hyperparameter

        # After finding tolerant lambda, reoptimize
        out = objective(lambd)

        return({
            'lambda': lambd,
            'beta': self.b0,
            "LL": out
        })

    # Inspection methods
    def get_Delta(self):
        pass

    def get_scores(self):
        pass

    def get_states(self):
        pass
