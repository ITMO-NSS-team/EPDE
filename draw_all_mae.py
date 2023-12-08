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
    DATA = "wave"

    if DATA == "wave":
        path = 'data_wave/'
        names = ["0", "2e-5", "2.5e-5", "3e-5", "3.2e-5", "3.47e-5"]
        core_values = [0.0087, 0.0094, 0.0449, 0.0453]
        core_colors = ["#385623", "#89BF65", "#C5E0B3", "#E2EFD9"]
        log_transform = False
        decimals = 4
        n_df = 2
    elif DATA == "burgers":
        path = 'data_burg/'
        names = ["0", "1e-5", "1.5e-5", "2e-5", "2.5e-5", "3e-5", "3.67e-5"]
        core_values = [0.0006, 0.0008, 0.001, 0.0014, 0.0132, 0.0137, 0.014, 0.0146]
        core_colors = ["#385623", "#538135", "#669D41", "#71AE48", "#89BF65", "#A8D08D", "#C5E0B3", "#E2EFD9"]
        log_transform = False
        decimals = 4
        n_df = 2
    elif DATA == "burgers_sindy":
        path = 'data_burg_sindy/'
        names = ["0", "0.001", "0.005", "0.01", "0.02", "0.03"]
        core_values = [8.0e-05, 2.0e-03, 6.0e-03, 1.3e-02, 2.1e-02, 1.0e-01, 2.0e-01]
        core_colors = ["#385623", "#43682A", "#538135", "#71AE48", "#C5E0B3", "#E2EFD9", "#F4F9F1"]
        log_transform = False
        decimals = [3, 1]
        n_df = 3
    elif DATA == "kdv_sindy":
        path = 'data_kdv_sindy/'
        names = ["0", "1e-5", "3.5e-5", "5.5e-5", "8e-5", "0.0001", "2.26e-4"]
        core_values = [3.0e-03, 8.0e-03, 1.0e-02, 1.4e-02, 1.9e-02, 2.1e-02, 4.4e-02, 1.6, 3.2]
        core_colors = ["#385623", "#43682A", "#5A8B39", "#669D41", "#89BF65", "#A8D08D", "#C5E0B3", "#E2EFD9", "#FDFEFC"]
        log_transform = True
        decimals = [3, 1]
        n_df = 3
    elif DATA == "kdv":
        path = 'data_kdv/'
        names = ["0", "0.001", "0.01", "0.07", "0.08", "0.09", "0.092"]
        core_values = [1.2e-05, 1.0e-04, 1.0e-03, 1.4e-03]
        core_colors = ["#385623", "#669D41", "#A8D08D", "#E2EFD9"]
        log_transform = False
        decimals = 4
        n_df = 2
    else:
        raise NameError('Unknown equation type')
    plot_mae(path, names, core_values, core_colors, decimals=decimals, n_df=n_df, log_transform=log_transform)
