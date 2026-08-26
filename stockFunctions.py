import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, mean_absolute_percentage_error


def conversion(y_train, stk_data):
    """
    Convert y_train into a DataFrame using the columns of stk_data.
    """
    actual_y_train = pd.DataFrame(
        index=range(len(y_train)),
        columns=stk_data.columns
    )

    for i in range(len(y_train)):
        actual_y_train.iloc[i] = y_train[i]

    return actual_y_train


def conversionSingle(y_train, stk_data):
    """
    Convert a single-column y_train into a DataFrame.
    stk_data can be a column name or a list of column names.
    """
    if isinstance(stk_data, str):
        columns = [stk_data]
    else:
        columns = stk_data

    actual_y_train = pd.DataFrame(
        index=range(len(y_train)),
        columns=columns
    )

    for i in range(len(y_train)):
        actual_y_train.iloc[i] = y_train[i]

    return actual_y_train


def graph(Actual, predicted, Actlabel, predlabel, title, Xlabel, ylabel):
    """
    Plot actual and predicted values.
    """
    plt.figure(figsize=(10, 5))

    plt.plot(
        Actual,
        label=Actlabel
    )

    plt.plot(
        predicted,
        label=predlabel
    )

    plt.title(title)
    plt.xlabel(Xlabel)
    plt.ylabel(ylabel)
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()


def rmsemape(y_Test, predicted_stock_price_test_ori):
    """
    Calculate and print RMSE and MAPE for the test set.
    """

    mse = mean_squared_error(
        y_Test,
        predicted_stock_price_test_ori
    )

    rmse = mse ** 0.5

    mape = mean_absolute_percentage_error(
        y_Test,
        predicted_stock_price_test_ori
    )

    print("RMSE-Testset:", rmse)
    print("MAPE-Testset:", mape)
