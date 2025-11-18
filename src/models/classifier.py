from abc import ABC, abstractmethod
from typing import Dict, List

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
)


class Classifier(ABC):
    @abstractmethod
    def train(self, *params) -> None:
        pass

    @abstractmethod
    def evaluate(self, *params) -> Dict[str, float]:
        pass

    @abstractmethod
    def predict(self, *params) -> np.ndarray:
        pass


class SklearnClassifier(Classifier):
    def __init__(self, estimator: BaseEstimator, features: List[str], target: str):
        self.clf = estimator
        self.features = features
        self.target = target

    def train(self, df_train: pd.DataFrame):
        self.clf.fit(df_train[self.features].values, df_train[self.target].values)

    def evaluate(self, df_test: pd.DataFrame) -> Dict[str, float]:
        
        
        y_true = df_test[self.target].values

        try:
            y_proba = self.predict(df_test)
            
            if y_proba.ndim == 2:
               
                y_proba = y_proba[:, 1]
        except Exception:
            if hasattr(self.clf, "predict_proba"):
                y_proba = self.clf.predict_proba(df_test[self.features].values)[:, 1]
            else:
                y_proba = None

      
        try:
            y_pred = self.clf.predict(df_test[self.features].values)
        except Exception:
            if y_proba is not None:
                y_pred = (y_proba >= 0.5).astype(int)
            else:
                raise RuntimeError("Unable to generate predictions from classifier.")

        metrics: Dict[str, float] = {}
        n_classes = len(np.unique(y_true))

        
        metrics["accuracy"] = float(accuracy_score(y_true, y_pred))
        
        avg = "binary" if n_classes == 2 else "macro"
        
        metrics["precision"] = float(precision_score(y_true, y_pred, average=avg, zero_division=0))
        metrics["recall"] = float(recall_score(y_true, y_pred, average=avg, zero_division=0))
        metrics["f1"] = float(f1_score(y_true, y_pred, average=avg, zero_division=0))

       

        return metrics

    def predict(self, df: pd.DataFrame) -> np.ndarray:
        
        if hasattr(self.clf, "predict_proba"):
            proba = self.clf.predict_proba(df[self.features].values)
           
            if proba.shape[1] == 2:
                return proba[:,1]
            return proba
        else:
            return self.clf.predict(df[self.features].values)