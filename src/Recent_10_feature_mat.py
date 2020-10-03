import pandas as pd
import numpy as np

def generate_features():
    movie_industry = pd.read_csv("../data/movie_industry.csv", encoding = "ISO-8859-1" )
    movie_industry = movie_industry[movie_industry.year >= 2007]
    directors2007 = np.unique(movie_industry.director.values)
    actors2007 = np.unique(movie_industry.star.values)
    
    df = pd.read_csv("../data/features.csv")
    condition = (df.agent.isin(directors2007) & (df.is_director == 1)) | (df.agent.isin(actors2007) & (df.is_director == 0))
    df = df[condition]
    df = df.loc[df.year>=2007]
    
    a = df[df.year == 2007].drop(columns = ["agent","is_director","id","year","time"]).to_numpy()
    b = df[df.year == 2008].drop(columns = ["agent","is_director","id","year","time"]).to_numpy()
    S = np.stack((a,b))
    
    for year in np.arange(2009, 2017):
        c = df[df.year == year].drop(columns = ["agent","is_director","id","year","time"]).to_numpy()
        S = np.concatenate((S, [c]))
    
    return S

def get_agents():
    movie_industry = pd.read_csv("../data/movie_industry.csv", encoding = "ISO-8859-1" )
    movie_industry = movie_industry[movie_industry.year >= 2007]
    directors2007 = np.unique(movie_industry.director.values)
    actors2007 = np.unique(movie_industry.star.values)
    
    df = pd.read_csv("../data/features.csv")
    condition = (df.agent.isin(directors2007) & (df.is_director == 1)) | (df.agent.isin(actors2007) & (df.is_director == 0))
    df = df[condition]
    df = df.loc[df.year>=2007]
    
    return df[df.year == 2007].agent.values