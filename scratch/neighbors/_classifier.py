import numpy as np
from ._base import NearestNeighbor

class KNeighborsClassifier(NearestNeighbor) :
    def __init__(
        self,
        n_neighbors = 5
    ) :
        super().__init__(
            n_neighbors=n_neighbors
        )

    def predict_prob(self, X_test) :
        neigh_ind = self._kneighbors(X_test)
        y_neigh = self.y[neigh_ind]
        X_test = np.array(X_test)
        classes = np.unique(self.y)
        neigh_prob = np.zeros((len(X_test), len(classes)))

        for i in range (len(y_neigh)):
            y_neigh_i = y_neigh[i]
            class_, count_ = np.unique(y_neigh_i, return_counts=True)
            prob_ = count_/len(y_neigh_i)

            #masukan setiap prob ke kelas yang ada
            for j in classes:
                neigh_prob[i, j] = prob_[j]

        return neigh_prob

    def predict(self, X_test) :
        classes = np.unique(self.y)
        neigh_prob = self.predict_prob(X_test)
        max_idx = neigh_prob.argmax(axis=1)

        y_pred = classes[max_idx]

        return y_pred