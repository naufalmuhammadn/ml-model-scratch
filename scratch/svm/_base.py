import numpy as np

class SVC :
    # SVM with SMO ref:https://cs229.stanford.edu/materials/smo.pdf
    
    def __init__(
        self,
        C = 1.0,
        tol = 1e-5,
        max_passes = 10     
    ) :
        self.C = C
        self.tol = tol
        self.max_passes = max_passes

    def fit(self, X, y) :
        X = np.array(X)
        y = np.array(y)
        n_samples, _ = X.shape

        self._initialize_parameters(X)
        passes = 0

        while (passes < self.max_passes) :
            num_changed_alphas = 0
            for i in range(n_samples) :
                X_i, y_i, a_i = X[i, :], y[i], self.alpha(i)

                E_i = self._calculate_E(X, y, X_i, y_i)

                cond_1 = y_i*E_i < -self.tol and a_i < self.C
                cond_2 = y_i*E_i > self.tol and a_i > 0

                if cond_1 or cond_2 :
                    j = self._generate_random(i, n_samples)

                    X_j, y_j, a_j = X[j,:], y[j], self.alpha[j]

                    E_j = self._calculate_E(X_j, y_j, a_j)

                    a_i_old, a_j_old = a_i, a_j

                    L, H = self._compute_L_H(y_i, y_j, a_i_old, a_j_old)

                    if L == H:
                        continue
                    
                    eta = 2*np.dot(X_i, X_j) - np.dot(X_i, X_i) - np.dot(X_j, X_j)

                    if eta >= 0 :
                        continue
                        
                    a_j = a_j - (y_j*(E_i-E_j))/eta

                    if a_j > H :
                        a_j = H
                    elif a_j < L :
                        a_j = L

                    if np.abs(a_j - a_j_old) < self.tol :
                        continue

                    a_i = a_i + (y_i*y_j)*(a_j_old-a_j)

                    b = self.intercept_

                    b_1 = b - E_i - (y_i*(a_i - a_i_old))*np.dot(X_i, X_i) - (y_j*(a_j - a_j_old))*np.dot(X_i, X_j)
                    b_2 = b - E_j - (y_i*(a_i - a_i_old))*np.dot(X_i, X_j) - (y_j*(a_j - a_j_old))*np.dot(X_j, X_j)

                    if a_i > 0 and a_i < self.C :
                        b = b_1
                    elif a_j > 0 and a_j < self.C :
                        b = b_2
                    else :
                        b = (b_1 + b_2)/2

                    self.alpha[i], self.alpha[j] = a_i, a_j
                    self.intercept_ = b
                    self.coef_ = np.dot(self.alpha*y, X)
                    num_changed_alphas += 1
            
            if num_changed_alphas == 0 :
                passes += 1
            else :
                passes = 0
    
    def predict(self, X) :
        f = np.dot(self.coef_, X) + self.intercept_

        if f >= 0 :
            y_pred = 1
        else :
            y_pred = 0

        return y_pred

    def _initialize_parameters(self, X) :
        n_samples, n_features = X.shape

        self.alpha = np.zeros(n_samples)
        self.coef_ = np.zeros(n_features)
        self.intercept_ = 0

    def _calculate_E(self, X, y, X_i, y_i) :
        f = self._calculate_F(X, y, X_i)
        return f - y_i
        
    def _calculate_F(self, X, y, X_i) :
        return np.dot(self.alpha*y, np.dot(X_i, X.T)) + self.intercept_

    def _generate_random(self, i, n_samples) :
        np.random.seed(10)
        j = np.random.choice(n_samples)

        while j == i :
            j = np.random.choice(n_samples)

        return j

    def _compute_L_H(self, y_i, y_j, a_i, a_j) :
        if y_i != y_j :
            L = max(0, a_j - a_i)
            H = min(self.C, self.C + a_j - a_i)
        else :
            L = max(0, a_i + a_j - self.C)
            H = min(self.C, a_i + a_j)

        return L, H