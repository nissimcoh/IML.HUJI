from __future__ import annotations
from typing import Tuple, NoReturn
from ...base import BaseEstimator
import numpy as np
from itertools import product

from ...metrics import misclassification_error


class DecisionStump(BaseEstimator):
    """
    A decision stump classifier for {-1,1} labels according to the CART algorithm

    Attributes
    ----------
    self.threshold_ : float
        The threshold by which the data is split

    self.j_ : int
        The index of the feature by which to split the data

    self.sign_: int
        The label to predict for samples where the value of the j'th feature is about the threshold
    """
    def __init__(self) -> DecisionStump:
        """
        Instantiate a Decision stump classifier
        """
        super().__init__()
        self.threshold_, self.j_, self.sign_ = None, None, None

    def _fit(self, X: np.ndarray, y: np.ndarray) -> NoReturn:
        """
        fits a decision stump to the given data

        Parameters
        ----------
        X : ndarray of shape (n_samples, n_features)
            Input data to fit an estimator for

        y : ndarray of shape (n_samples, )
            Responses of input data to fit to
        """
        allTuplesPlus = []
        allTuplesMinus = []
        for i in range(X.shape[1]):
            allTuplesPlus.append(self._find_threshold(X[:, i], y, 1))
            allTuplesMinus.append(self._find_threshold(X[:, i], y, -1))
        allTuples = allTuplesPlus + allTuplesMinus
        minError = 1
        index = 0
        for i in range(len(allTuples)):
            if(allTuples[i][1] < minError):
                minError = allTuples[i][1]
                index = i
        self.threshold_ = allTuples[index][0]
        if index >= len(allTuplesPlus):
            self.j_ = index - len(allTuplesPlus)
            self.sign_ = -1
        else:
            self.j_ = index
            self.sign_ = 1
        self.fitted_ = True

    def _predict(self, X: np.ndarray) -> np.ndarray:
        """
        Predict responses for given samples using fitted estimator

        Parameters
        ----------
        X : ndarray of shape (n_samples, n_features)
            Input data to predict responses for

        y : ndarray of shape (n_samples, )
            Responses of input data to fit to

        Returns
        -------
        responses : ndarray of shape (n_samples, )
            Predicted responses of given samples

        Notes
        -----
        Feature values strictly below threshold are predicted as `-sign` whereas values which equal
        to or above the threshold are predicted as `sign`
        """
        y = np.sign(self.sign_ * (X[:, self.j_] - self.threshold_))
        y[y == 0] = self.sign_
        return y

    def _find_threshold(self, values: np.ndarray, labels: np.ndarray, sign: int) -> Tuple[float, float]:
        """
        Given a feature vector and labels, find a threshold by which to perform a split
        The threshold is found according to the value minimizing the misclassification
        error along this feature

        Parameters
        ----------
        values: ndarray of shape (n_samples,)
            A feature vector to find a splitting threshold for

        labels: ndarray of shape (n_samples,)
            The labels to compare against

        sign: int
            Predicted label assigned to values equal to or above threshold

        Returns
        -------
        thr: float
            Threshold by which to perform split

        thr_err: float between 0 and 1
            Misclassificaiton error of returned threshold

        Notes
        -----
        For every tested threshold, values strictly below threshold are predicted as `-sign` whereas values
        which equal to or above the threshold are predicted as `sign`
        """
        minError = 1
        signedLabels = np.sign(labels)
        thr = values[0]
        for threshold in values:
            y = np.sign(sign * (values - threshold))
            y[y == 0] = sign
            mistakes = np.argwhere(y != signedLabels)
            thrErr = np.sum(np.abs(labels[mistakes])) / np.sum(np.abs(labels))
            if minError > thrErr:
                thr = threshold
                minError = thrErr
        return thr, minError

    def _loss(self, X: np.ndarray, y: np.ndarray) -> float:
        """
        Evaluate performance under misclassification loss function

        Parameters
        ----------
        X : ndarray of shape (n_samples, n_features)
            Test samples

        y : ndarray of shape (n_samples, )
            True labels of test samples

        Returns
        -------
        loss : float
            Performance under missclassification loss function
        """
        return misclassification_error(self.predict(X), y)
