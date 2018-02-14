import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.preprocessing import Imputer, LabelEncoder, StandardScaler
from sklearn.decomposition import PCA


def fix_missing_data_mean(data):
    """
    Fill missing data with mean value for column where it belong.
    :param data: matrix of features
    :return: matrix without missing data
    """

    # axis = 0, impute along columns
    imputer = Imputer(missing_values='NaN', strategy='mean', axis=0)
    # columns SES and MMSE have NaN values
    imputer = imputer.fit(data.values[:, 9:11])
    transformed = imputer.transform(data.values[:, 9:11])
    data["SES"] = transformed[:, 0]
    data["MMSE"] = transformed[:, 0]

    return data


def fix_missing_data_fill(data):
    """
    Fill missing data with last valid observation forward.
    :param data: Pandas DataFrame
    :return:
    """
    data = data.fillna(method='ffill')
    return data


def encode_categorical_data(data):
    """
    Encode categorical data; encode strings into numbers.

    :param data_x: matrix of independent features
    :param data_y: dependent variable array
    :return: matrix with encoded categorical data
    """

    # M/F column
    label_encoder_x = LabelEncoder()
    data['M/F'] = label_encoder_x.fit_transform(data.values[:, 5])

    # Group column
    label_encoder_y = LabelEncoder()
    data['Group'] = label_encoder_y.fit_transform(data.values[:, 2])

    return data


def scale_data(data_x):
    """
    Zero-center and normalize data.
    :param data_x:
    :return:
    """
    scaler_x = StandardScaler()
    data_x = scaler_x.fit_transform(data_x)
    return data_x


def pairplot(data):
    cols = ['MR Delay', 'Age', 'EDUC', 'SES', 'MMSE', 'CDR', 'eTIV', 'nWBV', 'ASF']
    sns.pairplot(data[cols])


def heat_map(data):
    corr_matrix = data.corr()
    sns.heatmap(corr_matrix, annot=True)


def feature_importance(X, y):
    # Build a forest and compute the feature importances
    # n_estimators = number of trees in forest
    forest = ExtraTreesClassifier(n_estimators=250,
                                  random_state=0)

    forest.fit(X, y)
    importances = forest.feature_importances_
    std = np.std([tree.feature_importances_ for tree in forest.estimators_],
                 axis=0)
    indices = np.argsort(importances)[::-1]

    # Print the feature ranking
    print("Feature ranking:")

    for f in range(X.shape[1]):
        print("%d. feature %d (%f)" % (f + 1, indices[f], importances[indices[f]]))

    # Plot the feature importances of the forest
    plt.figure()
    plt.title("Feature importances")
    plt.bar(range(X.shape[1]), importances[indices],
            color="r", yerr=std[indices], align="center")
    plt.xticks(range(X.shape[1]), indices)
    plt.xlim([-1, X.shape[1]])
    plt.show()


def pca(data):
    """
    Requires scaled data.
    :param data:
    :return:
    """
    # Number of components depends on how much variance we want to explain
    pca = PCA(n_components=4)
    # applying PCA to data set:
    # 1. fit PCA object to data set so that PCA can see how data object is structured and therefore
    # can extract some new independent variables that explain the most variance
    # 2. transform set
    data = pca.fit_transform(data)
    explained_variance = pca.explained_variance_ratio_
    # Calculating number of components based on explained variance.
    # Obtained result: 4 components explain 79,84% of variance
    for ev in explained_variance:
        print("%.2f" % ev)
    print('Explained variance of %d features: %.2f%%' % (len(data[0]), sum(explained_variance) * 100))

    return data


if __name__ == "__main__":
    # importing the dataset
    data = pd.read_csv('../data/oasis_longitudinal.csv')

    data = fix_missing_data_mean(data)
    data = encode_categorical_data(data)
    print(data.head())
    # remove unimportant features
    for x in ['Subject ID', 'MRI ID', 'Visit', 'Hand']:
        data.drop(x, axis=1, inplace=True)
    print(data.head())
    # split dataset
    X = data.iloc[:, 1:]
    y = data.iloc[:, 0]

    X = scale_data(X.astype('float'))
    feature_importance(X, y)
    # extract features
    X = pca(X)
