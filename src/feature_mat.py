import pandas as pd
import numpy as np
from scipy import sparse

def to_sparse(S):
    ret = []
    for i in range(S.shape[0]):
        ret.append(sparse.csr_matrix(S[i]))
    return np.array(ret)

def generate_features():
    df = pd.read_csv("../data/features.csv")
    a = df[df.year == 1986].drop(columns = ["agent","is_director","id","year","time"]).to_numpy()
    b = df[df.year == 1987].drop(columns = ["agent","is_director","id","year","time"]).to_numpy()
    S = np.stack((a,b))
    for year in np.arange(1988, 2017):
        c = df[df.year == year].drop(columns = ["agent","is_director","id","year","time"]).to_numpy()
        S = np.concatenate((S, [c]))
    return to_sparse(S)

def get_agents():
    df = pd.read_csv("../data/features.csv")
    return df[df.year == 1986].agents.values
