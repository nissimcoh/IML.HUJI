from IMLearn.learners.classifiers import Perceptron, LDA, GaussianNaiveBayes
from typing import Tuple
from utils import *
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from math import atan2, pi


def load_dataset(filename: str) -> Tuple[np.ndarray, np.ndarray]:
    """
    Load dataset for comparing the Gaussian Naive Bayes and LDA classifiers. File is assumed to be an
    ndarray of shape (n_samples, 3) where the first 2 columns represent features and the third column the class

    Parameters
    ----------
    filename: str
        Path to .npy data file

    Returns
    -------
    X: ndarray of shape (n_samples, 2)
        Design matrix to be used

    y: ndarray of shape (n_samples,)
        Class vector specifying for each sample its class

    """
    data = np.load(filename)
    return data[:, :2], data[:, 2].astype(int)


def run_perceptron():
    """
    Fit and plot fit progression of the Perceptron algorithm over both the linearly separable and inseparable datasets

    Create a line plot that shows the perceptron algorithm's training loss values (y-axis)
    as a function of the training iterations (x-axis).
    """

    for n, f in [("Linearly Separable", "linearly_separable.npy"), ("Linearly Inseparable", "linearly_inseparable.npy")]:
        # Load dataset
        filename = "C:\\Users\\Nissim\\IML.HUJI\\datasets\\" + f
        X, y = load_dataset(filename)
        # Fit Perceptron and record loss in each fit iteration
        def callableFunc(p: Perceptron, x: np.ndarray, i: int) -> None:
                losses.append(p.loss(x, y))
        losses = []
        perceptron = Perceptron(callback=callableFunc)
        perceptron.fit(X, y)
        # Plot figure of loss as function of fitting iteration
        fitting_iterations = [i for i in range(len(losses))]
        print(losses)
        print(fitting_iterations)
        go.Figure(go.Scatter(x=fitting_iterations, y=losses, mode='lines'),
                  layout=go.Layout(title="Figure of loss as function of fitting iteration " + n,
                                   xaxis_title="fitting iteration",
                                   yaxis_title="loss")).show()



def get_ellipse(mu: np.ndarray, cov: np.ndarray):
    """
    Draw an ellipse centered at given location and according to specified covariance matrix

    Parameters
    ----------
    mu : ndarray of shape (2,)
        Center of ellipse

    cov: ndarray of shape (2,2)
        Covariance of Gaussian

    Returns
    -------
        scatter: A plotly trace object of the ellipse
    """
    l1, l2 = tuple(np.linalg.eigvalsh(cov)[::-1])
    theta = atan2(l1 - cov[0, 0], cov[0, 1]) if cov[0, 1] != 0 else (np.pi / 2 if cov[0, 0] < cov[1, 1] else 0)
    t = np.linspace(0, 2 * pi, 100)
    xs = (l1 * np.cos(theta) * np.cos(t)) - (l2 * np.sin(theta) * np.sin(t))
    ys = (l1 * np.sin(theta) * np.cos(t)) + (l2 * np.cos(theta) * np.sin(t))

    return go.Scatter(x=mu[0] + xs, y=mu[1] + ys, mode="lines", marker_color="black")


def compare_gaussian_classifiers():
    """
    Fit both Gaussian Naive Bayes and LDA classifiers on both gaussians1 and gaussians2 datasets
    """

    for f in ["gaussian1.npy", "gaussian2.npy"]:
        # Load data
        filename = "C:\\Users\\Nissim\\IML.HUJI\\datasets\\" + f
        XTrain, yTrain = load_dataset(filename)

        # Fit models and predict over training set
        lda = LDA()
        gnb = GaussianNaiveBayes()

        lda.fit(XTrain, yTrain)
        gnb.fit(XTrain, yTrain)
        yLDAPred = lda.predict(XTrain)
        yGNBPredict = gnb.predict(XTrain)

        # Plot a figure with two suplots, showing the Gaussian Naive Bayes predictions on the left and LDA predictions
        # on the right. Plot title should specify dataset used and subplot titles should specify algorithm and accuracy
        # Create subplots
        from IMLearn.metrics import accuracy
        fig = make_subplots(cols=2, rows=1,
                            subplot_titles=(f + ",GNB Accuracy:" +
                                            str(accuracy(yTrain, yGNBPredict)),
                                                f + ",LDA Accuracy:" +
                                            str(accuracy(yTrain, yLDAPred))))
        # Add traces for data-points setting symbols and colors
        fig.add_trace(go.Scatter(mode="markers", x=XTrain[:, 0],y=XTrain[:, 1],
                                 marker=dict(color=yGNBPredict,
                                             symbol=yTrain)), row=1, col=1)
        fig.add_trace(go.Scatter(mode="markers", x=XTrain[:, 0],
                                 y=XTrain[:, 1], marker=dict(color=yLDAPred,
                                             symbol=yTrain)), row=1, col=2)

        # Add `X` dots specifying fitted Gaussians' means
        fig.add_trace(go.Scatter(mode="markers", x=gnb.mu_[:, 0],
                                 y=gnb.mu_[:, 1], marker=dict(color='black',
                                                              symbol='x')),
                      row=1, col=1)
        fig.add_trace(go.Scatter(mode="markers", x=lda.mu_[:, 0],
                                 y=lda.mu_[:,1], marker=dict(color='black',
                                                             symbol='x')),
                      row=1, col=2)

        # Add ellipses depicting the covariances of the fitted Gaussians
        for k in range(gnb.classes_.size):
            gnbCov = np.eye(XTrain.shape[1]) * gnb.vars_[k]
            fig.add_trace(get_ellipse(gnb.mu_[k], gnbCov), row=1, col=1)
            fig.add_trace(get_ellipse(lda.mu_[k], lda.cov_), row=1, col=2)
        fig.show()


if __name__ == '__main__':
    np.random.seed(0)
    run_perceptron()
    compare_gaussian_classifiers()
