import pandas as pd
from sklearn.model_selection import train_test_split

def read_and_split_data(path, y_column_name):
    df = pd.read_csv(path).dropna()
    X = df.drop(y_column_name, axis=1)
    y = df[y_column_name]
    return train_test_split(X, y, test_size=0.2, random_state=42)
