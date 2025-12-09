"""
Initialization module for data loading and preprocessing. This module exposes general functions that are used
throughout the project to load the dataset, separate features and labels, and divide the data into training,
validation, and test sets.
"""

from pandas import pandas as pd
from sklearn.model_selection import train_test_split

X_train, y_train, X_test, y_test, X_val, y_val = None, None, None, None, None, None

def load_data(file_path: str) -> pd.DataFrame:
    """Loads the dataset from a CSV file.
    
    Main function to load the dataset from a CSV file. It assumes the first row contains headers.
    The xtrain data consists on the tweets, while ytrain contains the labels (e.g., 'gender', 'ideology', etc.).

    Args:
        file_path (str): Path to the CSV file.

    Returns:
        pd.DataFrame: Loaded dataset.
    """
    data = pd.read_csv(file_path, header=0)
    return data

def divide_train_val_test(data: pd.DataFrame, train_size: float = 0.7, val_size: float = 0.15, test_size: float = 0.15):
    '''
    Divides the dataset into training, validation, and test sets.
    Train = 70%, Validation = 15%, Test = 15%
    Args:
        data (pd.DataFrame): The dataset to be divided.
        train_size (float): Proportion of the dataset to include in the training set.
        val_size (float): Proportion of the dataset to include in the validation set.
        test_size (float): Proportion of the dataset to include in the test set.
    '''
    train, temp = train_test_split(data, test_size=(val_size + test_size), random_state=42)
    val, test = train_test_split(temp, test_size=test_size/(val_size + test_size), random_state=42)
    return train, val, test

def separate_x_y_vectors(data: pd.DataFrame):
    '''
    Separates features and labels from the dataset.
    
    Args:
        data (pd.DataFrame): The dataset.
    '''
    X = data['tweet']
    y = data.iloc[:, 1:-1]
    return X, y

def get_data_splits(number_of_samples=None):
    '''
    Loads the dataset and prepares the training, validation, and test splits. If already cached, it loads from cache. If `number_of_samples` 
    is provided, it returns only that many samples from the training set.

    Args:
        number_of_samples (int, optional): Number of samples to return from the training set. If None, returns all samples.

    Returns:
        xtrain, ytrain, xvalidation, yvalidation, xtest, ytest
    '''
    global X_train, y_train, X_test, y_test, X_val, y_val

    if X_train is not None:
        if number_of_samples is not None:
            return X_train[:number_of_samples], y_train[:number_of_samples], X_val, y_val, X_test, y_test
        return X_train, y_train, X_val, y_val, X_test, y_test

    path = "Datasets/EvaluationData/politicES_phase_2_train_public.csv"
    data = load_data(path)
    
    train_data, val_data, test_data = divide_train_val_test(data)
    X_train, y_train = separate_x_y_vectors(train_data)
    X_val, y_val = separate_x_y_vectors(val_data)
    X_test, y_test = separate_x_y_vectors(test_data)

    if number_of_samples is not None:
        return X_train[:number_of_samples], y_train[:number_of_samples], X_val, y_val, X_test, y_test

    return X_train, y_train, X_val, y_val, X_test, y_test

label_name = 'gender'  # 'gender', 'ideology', etc.

traindata = load_data('Datasets/EvaluationData/politicES_phase_2_train_public.csv')
testdata = load_data('Datasets/EvaluationData/politicES_phase_2_test_public.csv')
validationdata = load_data('Datasets/PostEvaluationData/politicES_phase_2_test_codalab.csv')

xtrain = traindata['tweet']
ytrain = traindata[label_name]

xtest = testdata['tweet']
ytest = testdata[label_name]

xvalidation = validationdata['tweet']
yvalidation = validationdata[label_name]

print(xtest.shape)
