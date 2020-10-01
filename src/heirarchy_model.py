# heirarchy_model.py
# Description: Simulates heirarchy model based on Kawakatsu et al.
# based on https://github.com/PhilChodrow/prestige_reinforcement/blob/master/py/model.py
# with change to class structure and optimization method, underlying structure is mostly unchanged.

# Date: September 30, 2020
# Author: Marlin Figgins

# To do:
# Implement scoring based on statistical model
# Implement get_ methods for visualization
# Simulate heirarchies based on this model for Actor netowrks.

import numpy as np
from numba import jit


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


class heirarchy_model:
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

    def score(self, A, scoring):
        '''
        This function is used to score individuals in order to determine
        their probability of endorsing another agent.
        '''
        pass

    def simulate(self, lambd, scoring, steps):
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

        # initializing variables
        self.prob_mat = np.zeros((steps, n, n))
        self.A = np.zeros((steps.n, n, n))
        self.S = np.zeros((steps.n, n, n))

        for t in range(1, steps + 1):
            # compute scores
            s = self.score(A[t-1], scoring)
            self.S[t-1] = s

            # compute features
            # might be able to skip this as transformation might not be needed

            # compute prob_mat using scores
            prob_mat = np.exp(s)  # Taking R to [0. 1]
            prob_mat = prob_mat / prob_mat.sum(axis=1)[:, np.newaxis]

            # Compute update
            Delta[t-1] = deterministic_step(prob_mat)

            # update state
            A[t] = lambd*A[t-1] + (1 - lambd)*Delta[t-1]

            self.prob_mat[t-1] = prob_mat
            self.A[t-1] = A[t-1]
            return Delta

    def compute_state_from_deltas(self, lambd, A0=None):
        if A0 is None:
            A0 = self.Delta[0]

        A = np.zeros_like(self.Delta).astype(float)
        for t in range(1, self.steps + 1):
            A[t] = lambd*A[t-1] + (1 - lambd)*self.Delta[t-1]
        return A

    def compute_score(self):
        pass

    # Inference for lambda
    def compute_trajectory(self, lambd):
        '''
        Compute the trajectory A associated with a given lambda value
        '''
        self.A = self.state_from_deltas(lambd, self.A0)

    def compute_prob_mat(self):
        '''
        Compute probability matrix corresponding to the scores.
        '''
        prob_mat = np.exp(self.S)  # Taking R to [0. 1]
        prob_mat = prob_mat / prob_mat.sum(axis=2)[:, :, np.newaxis]
        self.prob_mat

    def likelihood(self):
        # We need only estimate lambd due to the fact we're using
        # a statistical model in our utility computations
        pass

    def optim(self, lambd0, alpha0=10 ** (-4), delta=10 ** (-4), tol=10 ** (-3), max_step=0.2):
        # We'll use a finite differences scheme to optimize this.

        def objective(lambd):
            self.compute_state_from_deltas(lambd)
            self.compute_score()

            # What should I compare this to? Is lambda just a hyperparameter?
            out = 1
            return out

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
