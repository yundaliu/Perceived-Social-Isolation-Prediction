import numpy as np
from collections import defaultdict
import itertools
from sklearn.decomposition import PCA


def return_n_components(data, threshold):
    pca = PCA(random_state=0).fit(data)
    explained_variance_ratio_ = pca.explained_variance_ratio_
    accumulated_ratio = np.array(list(itertools.accumulate(explained_variance_ratio_)))
    return np.sum(accumulated_ratio < threshold) + 1

    
class DecomposeEmbeddings():
    def __init__(self):
        pass
    
    def fit(self, X, feature_names, args):
        # only decompose embedding-based features
        n_features = len(feature_names)
        
        # group features
        self.regular_features = []
        self.embedding_features = defaultdict(list)
        for idx, name in enumerate(feature_names):
            if 'dim' not in name:
                self.regular_features.append(idx)
            else:
                name = name.split('_dim')[0]
                self.embedding_features[name].append(idx)
        
        # train pca on each group of embedding features
        self.pcas = {}
        for key, val in self.embedding_features.items():
            feats = X[:, val]
            
            # train pca
            n_components = return_n_components(feats, args['decomposition_threshold'])
            pca = PCA(n_components=n_components, random_state=0).fit(feats)
            
            # save
            self.pcas[key] = pca
        
        return self
    
    def transform(self, X):
        decomposed_X = []
        decomposed_X.append(X[:, self.regular_features])
        
        for key, indices in self.embedding_features.items():
            # extract features
            feats = X[:, indices]
            
            # decompose
            feats = self.pcas[key].transform(feats)
            
            # save
            decomposed_X.append(feats)
        
        return np.hstack(decomposed_X)
        
            
                
    