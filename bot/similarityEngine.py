import pandas as pd
import numpy as np
import logging
import pickle
from os import path

from sklearn.metrics.pairwise import cosine_similarity


class SimilarityEngine(object):
    """docstring for SimilarityEngine"""
    def __init__(self, modelpath):
        super(SimilarityEngine, self).__init__()
        logname = ("SurprisemeBot.log")
        self.log = logging.getLogger("similarityEngine")
        self.log.setLevel('INFO')
        formatter = logging.Formatter('%(asctime)-15s %(name)s '
                                      '[%(levelname)s] %(message)s')
        fh = logging.FileHandler(logname)
        fh.setFormatter(formatter)
        self.log.addHandler(fh)
        self.modelpath = modelpath
        # store = pd.HDFStore(modelpath)
        # df = store.get('df')
        with open(path.join(modelpath, "dois.pkl"), "rb") as infile:
            self.dois = pickle.load(infile)
        with open(path.join(modelpath, "doi2cidmapper.pkl"), "rb") as infile:
            self.doi2cidmapper = pickle.load(infile)
        with open(path.join(modelpath, "cid2doimapper.pkl"), "rb") as infile:
            self.cid2doimapper = pickle.load(infile)
        # self.vocabulary = df['word'].tolist()
        self.log.info("Size of dataset: %d" % len(self.dois))
        # self.mat = np.row_stack(df['vector'].tolist())
        with open(path.join(modelpath, "mat.pkl"), "rb") as infile:
            self.mat = pickle.load(infile)

    def get_related_papers(self, doi, n):
        i = self.dois.index(doi)
        cluster_id = self.mapper[doi]
        y = self.mat[i].reshape(1, -1)
        indices = np.array([self.dois.index(doi)
                            for doi in
                            self.cid2doimapper.get(cluster_id))
        candidates = [self.dois[i] for i in indices]
        X = self.mat[indices]
        sims = cosine_similarity(X, y).flatten()
        best_sims = sims.argsort()
        best_sims = list(best_sims[-10:-1])
        best_sims.reverse()
        results = []
        for i in best_sims:
            candidate = candidates[i]
            if not candidate == doi:
                results.append(candidates[i])
        return results[:n]

    def get_mixture_of_papers(self, dois, n):
        y_indices = [self.dois.index(doi) for doi in dois]
        cluster_ids = [self.mapper[doi] for doi in dois]
        y_vectors = [self.mat[i].reshape(1, -1) for i in y_indices]
        y = sum(y_vectors)
        x_indices = np.array([dois.index(doi)
                              for i in [14, 15]
                              for doi in cid2doimapper.get(i)])
        candidates = [self.dois[i] for i in x_indices]
        X = self.mat[x_indices]
        sims = cosine_similarity(X, y).flatten()
        best_sims = sims.argsort()
        best_sims = list(best_sims[-10:-1])
        best_sims.reverse()
        results = []
        for i in best_sims:
            candidate = candidates[i]
            if not candidate == doi:
                results.append(candidates[i])
        return results[:n]
