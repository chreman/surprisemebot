import pandas as pd
import numpy as np
import logging
import pickle
from os import path

import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import hdbscan

from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt

modelpath = "/home/chris/data/surpriseme/ldadf/"

with open(path.join(modelpath, "dois.pkl"), "rb") as infile:
    dois = pickle.load(infile)
with open(path.join(modelpath, "mat.pkl"), "rb") as infile:
    X = pickle.load(infile)


clusterer = hdbscan.HDBSCAN(min_cluster_size=60, min_samples=10,
                            cluster_selection_method='leaf')

labels = clusterer.fit_predict(X)
doi2cidmapper = {dois[i]: labels[i] for i in range(len(dois))}
cid2doimapper = {}
for doi, label in doi2cidmapper.items():
    if label in cid2doimapper:
        cid2doimapper[label].append(doi)
    else:
        cid2doimapper[label] = []
        cid2doimapper[label].append(doi)

with open(path.join(modelpath, "doi2cidmapper.pkl"), "wb") as outfile:
    pickle.dump(doi2cidmapper, outfile)
with open(path.join(modelpath, "cid2doimapper.pkl"), "wb") as outfile:
    pickle.dump(cid2doimapper, outfile)
