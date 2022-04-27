from IMLearn.utils import split_train_test
from IMLearn.learners.regressors import LinearRegression

from typing import NoReturn
import utils
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
import plotly.io as pio
pio.templates.default = "simple_white"


def load_data(filename: str):
    """
    Load house prices dataset and preprocess data.
    Parameters
    ----------
    filename: str
        Path to house prices dataset

    Returns
    -------
    Design matrix and response vector (prices) - either as a single
    DataFrame or a Tuple[DataFrame, Series]
    """
    dataFrame = pd.read_csv(filename)
    dataFrame.dropna()


    #remove alll the rare samples.
    dataFrame = dataFrame[dataFrame["price"] > 0]
    dataFrame = dataFrame[dataFrame["bedrooms"] >= 1]
    dataFrame = dataFrame[dataFrame["bathrooms"] >= 0.5]
    dataFrame = dataFrame[dataFrame["sqft_living"] >= 290]
    dataFrame = dataFrame[dataFrame["condition"] > 2]
    dataFrame = dataFrame[dataFrame["grade"] > 4.5]
    dataFrame = dataFrame[dataFrame["sqft_above"] < 5000]
    dataFrame = dataFrame[dataFrame["sqft_basement"] < 1700]
    dataFrame = dataFrame[dataFrame["view"] < 3]

    dataFrame.yr_renovated = np.log10(dataFrame.yr_renovated + 1)
    dataFrame.date = (pd.to_datetime(dataFrame.date[:]).dt.year - 2000) * 365 \
                     + pd.to_datetime(dataFrame.date[:]).dt.month * 30.4166 \
                     + pd.to_datetime(dataFrame.date[:]).dt.day
    dataFrame = dataFrame[dataFrame["date"] > 0]

    dataFrame = pd.get_dummies(dataFrame, prefix="zipcode num:",
                               columns=["zipcode"])

    prices = dataFrame.price
    dataFrame = dataFrame.drop(columns=["id", "lat", "long", "price",
                                        "sqft_living15", "sqft_lot15"])
    return dataFrame, prices

def feature_evaluation(X: pd.DataFrame, y: pd.Series, output_path: str = ".") -> NoReturn:
    """
    Create scatter plot between each feature and the response.
        - Plot title specifies feature name
        - Plot title specifies Pearson Correlation between feature and response
        - Plot saved under given folder with file name including feature name
    Parameters
    ----------
    X : DataFrame of shape (n_samples, n_features)
        Design matrix of regression problem

    y : array-like of shape (n_samples, )
        Response vector to evaluate against

    output_path: str (default ".")
        Path to folder in which plots are saved
    """
    sigmaY = np.std(y)
    for feature in X.iloc[:, :14]:
        sigmaX = np.std(X[feature])
        covXY = np.cov(X[feature], y)[0, 1]
        ro = covXY / (sigmaX * sigmaY)
        go.Figure(go.Scatter(x=X[feature], y=y, mode='markers'),
                         layout=go.Layout(title="The Person Correlation with "
                                                + str(feature) +
                                                "and price is: " + str(ro),
                                          xaxis_title=feature,
                                          yaxis_title="Price"))\
            .write_image(output_path + "\\" + feature + "PC.png")


if __name__ == '__main__':
    np.random.seed(0)



    # Question 1 - Load and preprocessing of housing prices dataset
    dataFrame, prices = load_data("C:\\Users\\Nissim\\IML.HUJI\\datasets\\house_prices.csv")


    # Question 2 - Feature evaluation with respect to response
    feature_evaluation(dataFrame, prices, "C:\\Users\\Nissim\\IML.HUJI\\Plots")


    # Question 3 - Split samples into training- and testing sets.
    dataFrameTrain, pricesTrain,  dataFrameTest, pricesTest = \
        split_train_test(dataFrame, prices)
    # Question 4 - Fit model over increasing percentages of the overall training data
    # For every percentage p in 10%, 11%, ..., 100%, repeat the following 10 times:
    #   1) Sample p% of the overall training data
    #   2) Fit linear model (including intercept) over sampled set
    #   3) Test fitted model over test set
    #   4) Store average and variance of loss over test set
    # Then plot average loss as function of training size with error ribbon of size (mean-2*std, mean+2*std)
    lossMean = []
    lossStd = []
    fractions = np.arange(10, 101)
    for p in fractions:
        loss = []
        for i in range(10):
            linearRegressionTester = LinearRegression()
            XTrain, yTrain, XTest, yTest =\
            split_train_test(dataFrame, prices, p*0.01)
            linearRegressionTester.fit(XTrain, yTrain)
            loss.append(linearRegressionTester.loss(dataFrameTest, pricesTest))
        lossMean.append(np.mean(loss))
        lossStd.append(np.std(loss))

    lossMean = np.array(lossMean)
    lossStd = np.array(lossStd)
    lossPlusTwo = lossMean + 2 * lossStd
    lossMinusTwo = lossMean - 2 * lossStd


    go.Figure((go.Scatter(x=fractions, y=lossMean, mode='markers+lines',
                          name='mean',
                         line=dict(dash='dash'), marker=dict(color='red')),
              go.Scatter(x=fractions, y=lossMinusTwo, mode='lines',
                         fill=None, line=dict(color="lightgrey"),
                         showlegend=False),
              go.Scatter(x=fractions, y=lossPlusTwo, mode='lines',
                         fill="tonexty", line=dict(color="lightgrey"),
                         showlegend=False)),
              layout=go.Layout(title="Mean of loss by p",
                               xaxis_title="Fraction",
                               yaxis_title="Mean")).show()
        # write_image("C:\\Users\\Nissim\\IML.HUJI\\Plots\\Q4.png")