import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.preprocessing import Imputer, LabelEncoder, OneHotEncoder


def fix_missing_data(data):
    """
    Taking care of missing data.

    :param data: matrix of features
    :return: matrix without missing data
    """
    # axis = 0, impute along columns
    imputer = Imputer(missing_values='NaN', strategy='mean', axis=0)
    imputer = imputer.fit(data[:, 6:7])
    data[:, 6:7] = imputer.transform(data[:, 6:7])
    return data


def encode_categorical_data(data_x, data_y):
    """
    Encode categorical data; encode strings into numbers.

    :param data_x: matrix of independent features
    :param data_y: dependent variable array
    :return: matrix with encoded categorical data
    """
    # hand column; all inputs are right hand, later will be removed
    data_x[:, 3] = 0
    label_encoder_x = LabelEncoder()
    # male/female column
    data_x[:, 2] = label_encoder_x.fit_transform(data_x[:, 2])

    label_encoder_y = LabelEncoder()
    data_y = label_encoder_y.fit_transform(data_y)

    return data_x, data_y


if __name__ == "__main__":
    # importing the dataset
    dataset = pd.read_csv('../data/oasis_longitudinal.csv')
    # take all rows & columns from 3rd
    X = dataset.iloc[:, 3:].values
    Y = dataset.iloc[:, 2].values

    X = fix_missing_data(X)
    X, Y = encode_categorical_data(X, Y)
