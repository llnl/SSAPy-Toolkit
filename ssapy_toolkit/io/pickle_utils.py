import pickle


def save_pickle(data, path):
    """Writes a dictionary to a pickle file."""
    with open(path, 'wb') as f:
        pickle.dump(data, f)


def read_pickle(path):
    """Reads a dictionary from a pickle file."""
    with open(path, 'rb') as f:
        return pickle.load(f)
