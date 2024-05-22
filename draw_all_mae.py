from drawbase.plot_mae import plot_mae

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
    DATA = "burgers"

    if DATA == "wave":
        path = 'data_wave/'
        # names = ["0", "2e-5", "2.5e-5", "3e-5", "3.2e-5", "3.47e-5"]
        names = ["0", "8.675e-6", "1.735e-5", "2.6025e-5", "3.47e-5"]
        core_values = [0.009, 0.044, 0.045]
        core_colors = ["#385623", "#71AE48", "#A8D08D"]
        log_transform = False
        decimals = 3
        n_df = 2
    elif DATA == "burgers":
        path = 'data_burg/'
        names = ["0", "9.175e-6", "1.835e-5", "2.7525e-5", "3.67e-5"]
        # core_values = [0.0006, 0.0017, 0.0127, 0.0135, 0.0143]
        # core_colors = ["#385623", "#43682A", "#71AE48", "#89BF65", "#A8D08D"]
        core_values = [0.001, 0.002, 0.013, 0.014, 0.67]
        core_colors = ["#385623", "#669D41", "#89BF65", "#A8D08D", "#FDFEFC"]
        log_transform = True
        decimals = [3, 2]
        n_df = 3
    elif DATA == "burgers_sindy":
        path = 'data_burg_sindy/'
        # names = ["0", "0.001", "0.005", "0.01", "0.02", "0.03"]
        # core_values = [8.0e-05, 2.0e-03, 6.0e-03, 1.3e-02, 2.1e-02, 1.0e-01, 2.0e-01]
        # core_colors = ["#385623", "#43682A", "#538135", "#71AE48", "#C5E0B3", "#E2EFD9", "#F4F9F1"]
        names = ["0", "0.0075", "0.015", "0.0225", "0.03"]
        core_values = [8.0e-05, 2.0e-03, 8.0e-03, 2.0e-02, 2.0e-01]
        core_colors = ["#385623", "#538135", "#669D41", "#C5E0B3", "#FDFEFC"]
        log_transform = False
        decimals = [3, 1]
        n_df = 3
    elif DATA == "kdv_sindy":
        path = 'data_kdv_sindy/'
        # names = ["0", "1e-5", "3.5e-5", "5.5e-5", "8e-5", "0.0001", "2.26e-4"]
        # names = ["0", "1e-5", "3.5e-5", "5.5e-5", "8e-5", "2.26e-4"]
        # core_values = [3.0e-03, 8.0e-03, 1.0e-02, 1.4e-02, 1.9e-02, 2.1e-02, 4.4e-02, 1.6, 3.1]
        # core_colors = ["#385623", "#43682A", "#5A8B39", "#669D41", "#89BF65", "#A8D08D", "#C5E0B3", "#E2EFD9", "#FDFEFC"]
        names = ["0", "2e-5", "4e-5", "6e-5", "8e-5"]
        core_values = [3.0e-03, 9.0e-03, 1.4e-02, 1.6e-02, 3.5e-02, 3.7e-02, 2.3e+00, 3.1e+00]
        core_colors = ["#385623", "#43682A", "#669D41", "#71AE48", "#A8D08D", "#C5E0B3", "#E2EFD9", "#FDFEFC"]
        log_transform = True
        decimals = [3, 1]
        n_df = 3
    elif DATA == "kdv":
        path = 'data_kdv/'
        # names = ["0", "0.001", "0.01", "0.07", "0.08", "0.09", "0.092"]
        # core_values = [2.1e-05, 1.0e-04, 1.0e-03, 1.6e-03]
        # core_colors = ["#385623", "#669D41", "#A8D08D", "#E2EFD9"]
        names = ["0", "0.023", "0.046", "0.069", "0.092"]
        core_values = [1.0e-05, 4.9e-05, 1.0e-04, 6.0e-04, 1.1e-03, 1.4e-03]
        core_colors = ["#385623", "#43682A", "#5A8B39", "#669D41", "#A8D08D", "#FDFEFC"]
        log_transform = False
        decimals = 4
        n_df = 2
    else:
        raise NameError('Unknown equation type')
    plot_mae(path, names, core_values, core_colors, decimals=decimals, n_df=n_df, log_transform=log_transform)
