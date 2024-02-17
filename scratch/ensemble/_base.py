import copy
import numpy as np

def _generate_sample_indices(seed, n_estimators, n_population, n_samples, bootstrap) :
    np.random.seed(seed)

    sample_indices = np.random.choice(n_population,
                                      size = (n_estimators,n_samples),
                                      replace = bootstrap)
    
    return sample_indices

def _generate_feature_indices(seed, n_estimators, n_population, n_features, bootstrap) :
    np.random.seed(seed)

    feature_indices = np.empty((n_estimators, n_features), dtype='int')
    for i in range(n_estimators) :
        feature_indices[i] = np.random.choice(n_population,
                                            n_features,
                                            replace=bootstrap)
        
        feature_indices[i].sort()
    
    return feature_indices

def _predict_ensemble(estimators, feature_indices, X):
    X = np.array(X).copy()
    n_samples = X.shape[0]

    n_estimators = len(estimators)

    y_preds = np.empty((n_estimators, n_samples))

    for i, estimator in enumerate(estimators) :
        X_ = X[:, feature_indices[i]]

        y_preds[i] = estimator.predict(X_)
    
    return y_preds

def _predict_aggregate(y_ensemble, aggregate_func, classes):
    n_estimators, n_samples = y_ensemble.shape

    y_pred = np.empty(n_samples)
    for i in range(n_samples) :
        y_samples = y_ensemble[:, i]

        y_pred[i] = aggregate_func(y_samples, classes)

    return y_pred

def majority_vote(y_samples, classes):
    y_pred_proba_col = []

    for cls_ in classes:
        cls_proba = np.sum(y_samples==cls_)/len(y_samples)
        y_pred_proba_col.append(cls_proba)

    max_proba_idx_col = np.argmax(y_pred_proba_col)
    
    return classes[max_proba_idx_col]


class BaseEnsemble:
    def __init__(
        self,
        estimator,
        n_estimators,
        max_features=None,
        random_state=None
    ):
        self.estimator = estimator
        self.n_estimators = n_estimators
        self.max_features = max_features
        self.random_state = random_state
        self.classes = list(sorted(set(y)))

    def fit(self, X, y):
        X = np.array(X).copy()
        y = np.array(y).copy()

        self.n_samples, self.n_features = X.shape

        self.estimators_ = [copy.deepcopy(self.estimator) for i in range(self.n_estimators)]

        MAX_INT = np.iinfo(np.int32).max
        if self.random_state is None :
            self.random_state = np.random.randint(0, MAX_INT)

        sample_indices = _generate_sample_indices(seed = self.random_state,
                                                  n_estimators=self.n_estimators,
                                                  n_population=self.n_samples,
                                                  n_samples=self.n_samples,
                                                  bootstrap=True)
                
        if isinstance(self.max_features, int) :
            max_features = self.max_features
        elif self.max_features == "sqrt" :
            max_features = int(np.sqrt(self.n_features))
        elif self.max_features == "log2" :
            max_features = int(np.log2(self.n_features))
        else :
            max_features = self.n_features
        
        self.feature_indices = _generate_feature_indices(seed=self.random_state,
                                                        n_estimators=self.n_estimators,
                                                        n_population=self.n_features,
                                                        n_features=max_features,
                                                        bootstrap=False)
        
        # Fit model
        for b in range(self.n_estimators):
            X_bts = X[:, self.feature_indices[b]]
            X_bts = X_bts[sample_indices[b], :]
            y_bts = y[sample_indices[b]]

            # Fitting
            estimator = self.estimators_[b]
            estimator.fit(X_bts, y_bts)
    
    def predict(self, X):
        y_pred_ensemble = _predict_ensemble(estimators=self.estimators_,
                                            feature_indices=self.feature_indices,
                                            X = X)
        
        y_pred = _predict_aggregate(y_pred_ensemble, majority_vote, self.classes)

        return y_pred