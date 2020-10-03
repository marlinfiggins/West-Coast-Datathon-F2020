# run_inference.py
# Description: Load in graph and covariates and conduct inference.
# Inputs: /data/graph.npy
# Outputs: /outputs/parameters.json

from Graph_Helper import load_timesteps
from hierarchy_model import hierarchy_model
from Recent_10_feature_mat import generate_features, generate_features_small
import json
import os

if __name__ == "__main__":
    # Load Delta as sparse array
    nodelist, Delta = load_timesteps(os.path.join(os.pardir, "data", "last_10_graph.npy"))

    # Load small subset of covariates
    cov = generate_features_small()
    # cov = generate_features()

    # Initialize hierarchy object with Delta and cov
    model = hierarchy_model(Delta=Delta[-10:], cov=cov[-10:]) # Restricting to most recent 10 years
    # model = hierarchy_model(Delta=Delta[Test_steps], cov=cov[Test_steps])

    # Optimize model
    parms = model.optim(lambd0 = 0.9)
    print(parms)

    with open('parameters.json', 'w') as file:
        json.dump(parms, file)

    with open(os.path.join(os.pardir, "data", "parameters.json"), "w") as file:
        json.dump(parms, file)
