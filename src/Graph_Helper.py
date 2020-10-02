import pandas as pd
import numpy as np
import seaborn as sb
import networkx as nx
import os
data_dir = ""

'''
sample usage for loading graph.npy:

from Graph_Helper import nodelist, delta
'''

'''

sample usage for loading graph.npy and converting the sparse arrays to numpy:

from Graph_Helper import nodelist, delta, convert_delta_to_np as convert

delta_as_numpy_arrays = convert(delta)

'''

# df = pd.read_csv(data_dir + 'movie_industry.csv', encoding="windows-1252")
# df2 = pd.read_csv(data_dir + 'the_oscar_award.csv')
# genome_scores = pd.read_csv(data_dir + 'genome-scores.csv')
# genome_tags = pd.read_csv(data_dir + 'genome-tags.csv')
# ratings = pd.read_csv(data_dir + 'ratings.csv')
# tags = pd.read_csv(data_dir + 'tags.csv')
# movies = pd.read_csv(data_dir + 'movies.csv')

class Graph:
    def __init__(self,
                 dta,
                 year,
                 director_column="director",
                 actor_column="star"):
        self.G = nx.Graph()
        directors = set(dta[director_column].values)
        actors = set(dta[actor_column].values)

        for director in directors:
            self.G.add_node((director, True))

        for actor in actors:
            self.G.add_node((actor, False))

        for director in directors:
            rows = dta[(dta["year"] == year) & (dta[director_column] == director)]
            for index in rows.index.values:
                self.G.add_edge((director, True), (rows.loc[index, actor_column], False),
                                weight=1)
    def to_numpy(self):
        return nx.adjacency_matrix(self.G, nodelist=self.G.nodes()).A

    def to_sparse(self):
        return nx.adjacency_matrix(self.G, nodelist=self.G.nodes())

    def nodes(self):
        return self.G.nodes()

def get_timesteps(dataset):
    lower = dataset["year"].values[0]
    upper = dataset["year"].values[-1]
    nodelist = Graph(dataset, lower).nodes()
    ret = []
    for i in range(lower, upper + 1):
        ret.append(Graph(dataset, i).to_numpy())
    return np.array([nodelist, np.array(ret)])

def save_timesteps(dataset, file_name):
    np.save(data_dir + file_name, get_timesteps(dataset))

def convert_delta_to_np(delta):
    return np.array([entry.A for entry in delta])

def load_timesteps(file_name):
    return np.load(file_name, allow_pickle=True)

nodelist, delta = load_timesteps(os.path.join(os.pardir, "data", "graph.npy"))
