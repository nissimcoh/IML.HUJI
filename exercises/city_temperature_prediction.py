from pygments.lexers import go

import IMLearn.learners.regressors.linear_regression
from IMLearn.learners.regressors import PolynomialFitting, polynomial_fitting
from IMLearn.utils import split_train_test

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.io as pio
pio.templates.default = "simple_white"


def load_data(filename: str) -> pd.DataFrame:
    """
    Load city daily temperature dataset and preprocess data.
    Parameters
    ----------
    filename: str
        Path to house prices dataset

    Returns
    -------
    Design matrix and response vector (Temp)
    """
    dataFrame = pd.read_csv(filename, parse_dates=["Date"])
    dataFrame.dropna()

    # remove alll the rare and invalid samples.
    dataFrame = dataFrame[dataFrame["Day"] >= 1]
    dataFrame = dataFrame[dataFrame["Day"] <= 31]
    dataFrame = dataFrame[dataFrame["Month"] >= 1]
    dataFrame = dataFrame[dataFrame["Month"] <= 12]
    dataFrame = dataFrame[dataFrame["Year"] >= 1995]
    dataFrame = dataFrame[dataFrame["Year"] <= 2020]
    dataFrame = dataFrame[dataFrame["Temp"] >= -10]

    dataFrame["DayOfYear"] = dataFrame.Date.dt.day_of_year

    # dataFrame = pd.get_dummies(dataFrame, prefix="in ",columns=["City"])
    # dataFrame = dataFrame.drop(columns=["Country", "City"])

    # dataFrame.to_csv("C:\\Users\\Nissim\\IML.HUJI\\Plots\\12345.csv")
    # print(dataFrame.DayOfYear)

    return dataFrame


if __name__ == '__main__':
    np.random.seed(0)


    # Question 1 - Load and preprocessing of city temperature dataset
    dataFrame = load_data("C:\\Users\\Nissim\\IML.HUJI\\datasets\\City_Temperature.csv")

    # Question 2 - Exploring data for specific country
    IsraelDataFrame = dataFrame[dataFrame["Country"] == "Israel"]
    IsraelDataFrame["Year"] = IsraelDataFrame["Year"].astype(str)
    IsraelTempAndDay = IsraelDataFrame.drop(columns=["Date", "Day", "City",
                                                     "Year", "Month", "Country"])
    fig = px.scatter(IsraelDataFrame, x="DayOfYear",y="Temp", color="Year", height=500)
    # fig.show()
    monthFig = px.bar(IsraelDataFrame.groupby("Month").agg('std'),y="Temp", height=500)
    # monthFig.show()

    # Question 3 - Exploring differences between countries
    groupedDataFrame = dataFrame.groupby(["Country", "Month"], as_index=False).agg(mean=("Temp", "mean"), std=("Temp", "std")).reset_index()
    monthFigAllCountries = px.line(groupedDataFrame,x="Month",y="mean",color="Country",
                                  height=500, error_y="std")
    # monthFigAllCountries.show()


    # Question 4 - Fitting model for different values of `k`
    degrees = np.arange(1, 11)

    trainDataFrame = IsraelDataFrame.sample(frac=0.75)
    testDataFrame = IsraelDataFrame.drop(trainDataFrame.index)
    loss = []
    for k in degrees:
        PF = PolynomialFitting(k)
        lossVal = PF.fit(trainDataFrame.DayOfYear, trainDataFrame.Temp)\
            .loss(testDataFrame.DayOfYear, testDataFrame.Temp)
        loss.append(lossVal)
    loss = np.array(loss)
    print(loss)
    polyFig = px.bar(x=degrees, y=loss, height=500)
    polyFig.update_xaxes(title_text="Degree")
    polyFig.update_yaxes(title_text="Loss")
    # polyFig.show()

    # Question 5 - Evaluating fitted model on different countries
    PF = PolynomialFitting(5)
    PF.fit(trainDataFrame.DayOfYear, trainDataFrame.Temp)

    countries = ["South Africa", "The Netherlands", "Jordan"]
    lossOtherCountries = []
    for country in countries:
        countryDataFrame = dataFrame[dataFrame["Country"] == country]
        lossOtherCountries.append(PF.loss(countryDataFrame.DayOfYear, countryDataFrame.Temp))
    lossOtherCountries = np.array(lossOtherCountries)
    polyFig2 = px.bar(x=countries, y=lossOtherCountries, height=500)
    polyFig2.update_xaxes(title_text="Country")
    polyFig2.update_yaxes(title_text="Loss")
    polyFig2.show()