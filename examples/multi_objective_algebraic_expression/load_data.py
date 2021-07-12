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
    try:
        return load_data[idx]
    except IndexError:
        raise IndexError('There is no data with this index')