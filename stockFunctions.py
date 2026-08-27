import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, mean_absolute_percentage_error

def split_sequences(sequences, n_steps_in, n_steps_out):

    X, y = list(), list()

    for i in range(len(sequences)):
        end_ix = i + n_steps_in
        out_end_ix = end_ix + n_steps_out
        if out_end_ix > len(sequences):
            break
        seq_x, seq_y = sequences[i:end_ix, :], sequences[end_ix:out_end_ix, :]

        X.append(seq_x)
        y.append(seq_y)

    from numpy import array

    return array(X), array(y)
    
    
def conversion(y_train, data):
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


def conversionSingle(y_train, data):
    """
    Convert a single-column y_train into a DataFrame.
    stk_data can be a column name or a list of column names.
    """
    if isinstance(data, str):
        columns = [data]
    else:
        columns = data

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
