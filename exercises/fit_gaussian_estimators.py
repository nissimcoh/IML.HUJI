from IMLearn.learners import UnivariateGaussian, MultivariateGaussian
import numpy as np
import plotly.graph_objects as go
import plotly.io as pio
pio.templates.default = "simple_white"


def test_univariate_gaussian():
    # Question 1 - Draw samples and print fitted model
    X = np.random.normal(10, 1, 1000)
    univariate = UnivariateGaussian()
    univariate.fit(X)
    print((univariate.mu_, univariate.var_))

    # # Question 2 - Empirically showing sample mean is consistent
    univariate = UnivariateGaussian()
    samplesNum = np.linspace(10,1000, 100)
    expectations = np.array([np.abs(10 -univariate.fit(X[:(i + 1) * 10]).mu_) for i in range(100)])

    go.Figure(go.Scatter(x=samplesNum, y=expectations, mode='markers+lines'),
              layout=go.Layout(
                  title=r"$\text{Q2 - Estimator dist from mean by samples}$",
                  xaxis_title="$\\text{number of samples}$",
                  yaxis_title="$\\text{absolute dif of estimator from mean}$",
                  height=400)).show()
    #
    #
    # # Question 3 - Plotting Empirical PDF of fitted model
    Xsorted = np.sort(X)
    afterPDF = univariate.pdf(Xsorted)
    go.Figure(go.Scatter(x=Xsorted, y=afterPDF, mode='markers+lines'),
              layout=go.Layout(
                  title=r"$\text{Q3 - PDF on samples}$",
                  xaxis_title="$\\text{origin sample}$",
                  yaxis_title="$\\text{sample after PDF}$",
                  height=400)).show()


def test_multivariate_gaussian():
    # Question 4 - Draw samples and print fitted model
    expectations = np.array([0, 0, 4, 0])
    covariance = np.matrix([[1, 0.2, 0, 0.5], [0.2, 2, 0, 0], [0, 0, 1, 0],
                            [0.5, 0, 0, 1]])
    X = np.random.multivariate_normal(expectations, covariance, 1000)
    multivariate = MultivariateGaussian()
    multivariate.fit(X)
    print("expectations: \n", multivariate.mu_)
    print("covariance matrix: \n", multivariate.cov_)


    # Question 5 - Likelihood evaluation
    F1 = np.linspace(-10, 10, 200)
    F3 = np.linspace(-10, 10, 200)
    LLvals = np.zeros((200, 200))
    i = 0
    for f1 in F1:
        for f3 in F3:
            expectations = np.array([f1, 0, f3, 0])
            LLvals[i // 200, i % 200] = \
                multivariate.log_likelihood(expectations, covariance, X)
            i += 1
    go.Figure(data=go.Heatmap(x=F1, y=F3, z=LLvals),
        layout=go.Layout(
            title=r"$\text{Log likelihood heatmap}$",
            xaxis_title="$\\text{f1}$", yaxis_title="$\\text{f3}$",
            width=500, height=500)).show()

    # Question 6 - Maximum likelihood
    print("f1 = ", F1[np.where(LLvals == np.amax(LLvals))[1]][0],
          "f3 = ", F3[np.where(LLvals == np.amax(LLvals))[0]][0])


if __name__ == '__main__':
    np.random.seed(0)
    test_univariate_gaussian()
    test_multivariate_gaussian()
