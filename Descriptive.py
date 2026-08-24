# -*- coding: utf-8 -*-

import pandas as pd
import numpy as np


class DescriptiveAnalysis:

    @staticmethod
    def qual_quan(data):

        qual = []
        quan = []

        for columnName in data.columns:

            if data[columnName].dtypes == 'O':
                qual.append(columnName)
            else:
                quan.append(columnName)

        return quan, qual


    @staticmethod
    def freqtable(columnName, data):

        Frequency = pd.DataFrame(
            columns=[
                "Unique_Values",
                "Frequency",
                "Relative_Frequency",
                "Cumulative_Frequency"
            ]
        )

        Frequency["Unique_Values"] = data[columnName].value_counts().index

        Frequency["Frequency"] = data[columnName].value_counts().values

        Frequency["Relative_Frequency"] = (
            Frequency["Frequency"] / len(data)
        )

        Frequency["Cumulative_Frequency"] = (
            Frequency["Relative_Frequency"].cumsum()
        )

        return Frequency


    @staticmethod
    def Univariate(quan, data):

        Descriptive = pd.DataFrame(
            index=[
                "Mean",
                "Median",
                "Mode",
                "Q1:25%",
                "Q2:50%",
                "Q3:75%",
                "Q4:100%",
                "%99",
                "IQR",
                "1.5rule",
                "Lesser",
                "Greater",
                "Max",
                "Min",
                "Skewness",
                "Kurtosis",
                "Variance",
                "Std"
            ],
            columns=quan
        )

        for columnName in quan:

            Descriptive.loc["Mean", columnName] = (
                data[columnName].mean()
            )

            Descriptive.loc["Median", columnName] = (
                data[columnName].median()
            )

            Descriptive.loc["Mode", columnName] = (
                data[columnName].mode()[0]
            )

            Descriptive.loc["Q1:25%", columnName] = (
                data[columnName].quantile(0.25)
            )

            Descriptive.loc["Q2:50%", columnName] = (
                data[columnName].quantile(0.50)
            )

            Descriptive.loc["Q3:75%", columnName] = (
                data[columnName].quantile(0.75)
            )

            Descriptive.loc["Q4:100%", columnName] = (
                data[columnName].max()
            )

            Descriptive.loc["%99", columnName] = (
                np.percentile(data[columnName], 99)
            )

            Descriptive.loc["IQR", columnName] = (
                Descriptive.loc["Q3:75%", columnName]
                - Descriptive.loc["Q1:25%", columnName]
            )

            Descriptive.loc["1.5rule", columnName] = (
                1.5 * Descriptive.loc["IQR", columnName]
            )

            Descriptive.loc["Lesser", columnName] = (
                Descriptive.loc["Q1:25%", columnName]
                - Descriptive.loc["1.5rule", columnName]
            )

            Descriptive.loc["Greater", columnName] = (
                Descriptive.loc["Q3:75%", columnName]
                + Descriptive.loc["1.5rule", columnName]
            )

            Descriptive.loc["Max", columnName] = (
                data[columnName].max()
            )

            Descriptive.loc["Min", columnName] = (
                data[columnName].min()
            )

            Descriptive.loc["Skewness", columnName] = (
                data[columnName].skew()
            )

            Descriptive.loc["Kurtosis", columnName] = (
                data[columnName].kurtosis()
            )

            Descriptive.loc["Variance", columnName] = (
                data[columnName].var()
            )

            Descriptive.loc["Std", columnName] = (
                data[columnName].std()
            )

        return Descriptive