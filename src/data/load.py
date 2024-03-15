import pandas as pd

def load_data(file_path):
    """Load dataset from a specified file path."""
    return pd.read_csv(file_path)

def load_pickle(file_path):
    """Load a pickle file."""
    import pickle
    with open(file_path, 'rb') as f:
        data = pickle.load(f)
    return data