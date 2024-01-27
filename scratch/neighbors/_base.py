import numpy as np

class NearestNeighbor:
    def __init__(
        self,
        n_neighbors=5
    ):
        self.n_neighbors = n_neighbors

    # Membuat predict 
    def _kneighbors(self, X_test):
        # Convert ke numpy array
        X_test = np.array(X_test)

        # Inisialisasi
        n_queries = len(X_test)
        n_samples = len(self.X)
        list_dist = np.zeros((n_queries, n_samples))

        #Cari jarak
        for i in range (n_queries) :
            for j in range (n_samples) :
                X_test_i = X_test[i]
                X_j = self.X[j]
                
                dist_ij = np.linalg.norm(X_test_i - X_j)
                list_dist[i,j] = dist_ij

        #Cari tetangga
        neigh_ind = np.argsort(list_dist)[:, :self.n_neighbors]

        return neigh_ind

        
    # Cuma menyimpan data, bukan modeling
    def fit(self, X, y) :
        self.X = np.array(X)
        self.y = np.array(y)