import seaborn as sns
from matplotlib import rcParams
from drawbase.plot_time import plot_time
rcParams.update({'figure.autolayout': True})
sns.set(style="whitegrid", color_codes=True)

'''

    The variable DATA has the following possible states:
    wave;
    burgers;
    kdv;
    kdv_sindy;
    burgers_sindy;
    which corresponds to 5 input datasets and equations

'''
if __name__ == '__main__':
    DATA = "wave"

    n_df = 2
    if DATA == "wave":
        path = 'data_wave/'
        names = ["0", "2e-5", "2.5e-5", "3e-5", "3.2e-5", "3.47e-5"]
    elif DATA == "burgers":
        path = 'data_burg/'
        names = ["0", "1e-5", "1.5e-5", "2e-5", "2.5e-5", "3e-5", "3.67e-5"]
    elif DATA == "burgers_sindy":
        path = 'data_burg_sindy/'
        names = ["0", "0.001", "0.005", "0.01", "0.02", "0.03"]
    elif DATA == "kdv_sindy":
        path = 'data_kdv_sindy/'
        names = ["0", "1e-5", "3.5e-5", "5.5e-5", "8e-5", "0.0001", "2.26e-4"]
    elif DATA == "kdv":
        path = 'data_kdv/'
        names = ["0", "0.001", "0.01", "0.07", "0.08", "0.09", "0.092"]
    else:
        raise NameError('Unknown equation type')
    plot_time(path, names, n_df)
