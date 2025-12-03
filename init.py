from pandas import pandas as pd

def load_data(file_path: str) -> pd.DataFrame:
    """Loads the dataset from a CSV file.
    
    Main funtion to load the dataset from a CSV file. It assumes the first row contains headers.

    Args:
        file_path (str): Path to the CSV file.

    Returns:
        pd.DataFrame: Loaded dataset.
    """
    data = pd.read_csv(file_path, header=0)
    return data

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