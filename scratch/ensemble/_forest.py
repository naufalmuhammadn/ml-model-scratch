import numpy as np
from ._base import BaseEnsemble
from sklearn.tree import DecisionTreeClassifier

class RandomForestClassifier(BaseEnsemble):
    def __init__(
        self,
        estimator=None,
        n_estimators=5,
        max_features='sqrt',
        random_state=None
    ):
        if estimator is None:
            estimator = DecisionTreeClassifier()
        
        self.aggregate_function = self.majority_vote

        super().__init__(
            estimator=estimator,
            n_estimators=n_estimators,
            max_features=max_features,
            random_state=random_state
        )