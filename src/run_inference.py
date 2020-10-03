# run_inference.py
# Description: Load in graph and covariates and conduct inference.
# Inputs: /data/graph.npy
# Outputs: /outputs/parameters.json

from Graph_Helper import load_timesteps
from hierarchy_model import hierarchy_model
from feature_mat import *
import json
import os

if __name__ == "__main__":
    # Load Delta as sparse array
    nodelist, Delta = load_timesteps(os.path.join(os.pardir, "data", "graph.npy"))

    # Load small subset of covariates
    cov = generate_features_small()

    # Initialize hierarchy object with Delta and cov
    model = hierarchy_model(Delta=Delta, cov=cov)

    # Optimize model
    parms = model.optim(lambd0 = 0.1)
    print(parms)

    with open('parameters.json', 'wb') as file:
        json.dump(parms, file)

    with open(os.path.join(os.pardir, "data", "parameters.json", "wb")) as file:
        json.dump(parms, file)
