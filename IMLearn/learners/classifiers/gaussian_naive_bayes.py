from typing import NoReturn
from IMLearn.base import BaseEstimator
import numpy as np
from numpy.linalg import det, inv


class GaussianNaiveBayes(BaseEstimator):
    """
    Gaussian Naive-Bayes classifier
    """
    def __init__(self):
        """
        Instantiate a Gaussian Naive Bayes classifier

        Attributes
        ----------
        self.classes_ : np.ndarray of shape (n_classes,)
            The different labels classes. To be set in `GaussianNaiveBayes.fit`

        self.mu_ : np.ndarray of shape (n_classes,n_features)
            The estimated features means for each class. To be set in `GaussianNaiveBayes.fit`

        self.vars_ : np.ndarray of shape (n_classes, n_features)
            The estimated features variances for each class. To be set in `GaussianNaiveBayes.fit`

        self.pi_: np.ndarray of shape (n_classes)
            The estimated class probabilities. To be set in `GaussianNaiveBayes.fit`
        """
        super().__init__()
        self.classes_, self.mu_, self.vars_, self.pi_ = None, None, None, None

    def _fit(self, X: np.ndarray, y: np.ndarray) -> NoReturn:
        """
        fits a gaussian naive bayes model

        Parameters
        ----------
        X : ndarray of shape (n_samples, n_features)
            Input data to fit an estimator for

        y : ndarray of shape (n_samples, )
            Responses of input data to fit to
        """

        self.classes_, mk = np.unique(y, return_counts=True)
        if len(X.shape) != 1:
            self.mu_ = np.zeros((self.classes_.size, X.shape[1]))
            self.vars_ = np.zeros((self.classes_.size, X.shape[1]))
        else:
            self.mu_ = np.zeros((self.classes_.size, 1))
            self.vars_ = np.zeros((self.classes_.size, 1))
        self.pi_ = mk / y.size
        for k in range(self.classes_.size):
            indexes = np.where(y == self.classes_[k])[0]
            self.mu_[k] += (1 / mk[k]) * sum(X[i] for i in indexes)
            x = np.array([X[i] for i in indexes])
            self.vars_[k] = x.var(axis=0, ddof=1)
        self.fitted_ = True


    def _predict(self, X: np.ndarray) -> np.ndarray:
        """
        Predict responses for given samples using fitted estimator

        Parameters
        ----------
        X : ndarray of shape (n_samples, n_features)
            Input data to predict responses for

        Returns
        -------
        responses : ndarray of shape (n_samples, )
            Predicted responses of given samples
        """
        return np.array(np.argmax(self.likelihood(X), axis=1))

    def likelihood(self, X: np.ndarray) -> np.ndarray:
        """
        Calculate the likelihood of a given data over the estimated model

        Parameters
        ----------
        X : np.ndarray of shape (n_samples, n_features)
            Input data to calculate its likelihood over the different classes.

        Returns
        -------
        likelihoods : np.ndarray of shape (n_samples, n_classes)
            The likelihood for each sample under each of the classes

        """
        if not self.fitted_:
            raise ValueError("Estimator must first be fitted before calling `likelihood` function")

        likelihood = []
        for i in range(self.classes_.size):
            cov = np.eye(X.shape[1]) * self.vars_[i]
            pdf = (det(2 * np.pi * cov) ** -0.5) * \
                  np.exp(np.diag(-0.5 * (X - self.mu_[i]) @ inv(cov) @
                                 (X - self.mu_[i]).T))
            likelihood.append(pdf * self.pi_[i])
        return np.array(likelihood).T

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
        from IMLearn.metrics import misclassification_error
        return misclassification_error(y, self._predict(X))

