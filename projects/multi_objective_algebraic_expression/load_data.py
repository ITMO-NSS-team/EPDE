import pickle


load_data = []

path = r'.\data\ts_samples.pkl'

with open(path, 'rb') as file:
    while True:
        try:
            load_data.append(pickle.load(file))
        except:
            break


def get_data(idx: int) -> dict:
    """
    Retrieves a specific data entry from the dataset.
    
    This function allows access to individual data points within the loaded dataset, 
    enabling further analysis or processing of specific instances. It is essential 
    for examining particular scenarios or events represented in the data.
    
    Args:
        idx (int): The index of the desired data entry in the dataset.
    
    Returns:
        dict: The data entry at the specified index.
    
    Raises:
        IndexError: If the provided index is outside the valid range of the dataset.
    """
    try:
        return load_data[idx]
    except IndexError:
        raise IndexError('There is no data with this index')