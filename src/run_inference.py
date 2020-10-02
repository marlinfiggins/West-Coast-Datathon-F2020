# run_inference.py
# Description: Load in graph and covariates and conduct inference.
# Inputs: /data/graph.npy
# Outputs: /outputs/parameters.json

import os
from Graph_Helper import load_timesteps
from hierarchy_model import hierarchy_model

import json

if __name__ == "__main__":
    # Load Delta as sparse array
    nodelist, Delta = load_timesteps(os.path.join(os.pardir, "data", "graph.npy"))

    # Load covariates
    cov = None

    # Initialize hierarchy object with Delta and cov
    model = hierarchy_model(Delta=Delta, cov=cov)

    # Optimize model
    parms = model.optim()

    with open('../outputs/parameters.json', 'wb') as file:
        json.dump(parms, file)
