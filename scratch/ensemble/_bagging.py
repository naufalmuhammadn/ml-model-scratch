import numpy as np
from ._base import BaseEnsemble
from sklearn.tree import DecisionTreeClassifier

def majority_vote(y_samples, classes):
    y_pred_proba_col = []

    for cls_ in classes:
        cls_proba = np.sum(y_samples==cls_)/len(y_samples)
        y_pred_proba_col.append(cls_proba)

    max_proba_idx_col = np.argmax(y_pred_proba_col)
    
    return classes[max_proba_idx_col]

class BaggingClassifier(BaseEnsemble):
    def __init__(
        self,
        estimator=None,
        n_estimators=10,
        random_state=None
    ):
        if estimator is None:
            estimator = DecisionTreeClassifier()
        
        self.aggregate_function = majority_vote

        super().__init__(
            estimator=estimator,
            n_estimators=n_estimators,
            random_state=random_state
        )
