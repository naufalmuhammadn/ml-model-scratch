import numpy as np
from ._base import NearestNeighbor

class KNeighborsRegressor(NearestNeighbor) :
    def __init__(
        self,
        n_neighbors = 5
    ) :
        super().__init__(
            n_neighbors=n_neighbors
        )

    def predict(self, X_test):
        neigh_ind = self._kneighbors(X_test)
        
        y_neigh = self.y[neigh_ind]

        y_pred = y_neigh.mean(axis=1)
        
        return y_pred