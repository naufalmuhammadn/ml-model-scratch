import numpy as np

class LinearRegression:
    def __init__ (
        self,
        fit_intercept=True
    ):
        self.fit_intercept=fit_intercept

    def fit(self, X, y) :
        X = np.array(X)
        y = np.array(y)
        
        n_samples, n_features = X.shape

        if self.fit_intercept:
            A = np.column_stack((X, np.ones(n_samples)))
        else:
            A = X
        
        theta = np.linalg.inv(A.T @ A) @ A.T @ y
        
        # Extracting coef and intercept
        if self.fit_intercept:
            self.coef_ = theta[:-1]
            self.intercept_ = theta[-1]
        else:
            self.coef_ = theta
            self.intercept_ = 0.0

    def predict(self, X_test) :
        y_pred = X_test @ self.coef_ + self.intercept_

        return y_pred