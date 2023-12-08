from drawbase.preprocess_df import melt_count_time

path = 'data_burg_sindy/'
names = ["0", "0.001", "0.005", "0.01", "0.02", "0.03"]
n_df = 3
dfall = melt_count_time(path, names, n_df)
time = dfall[dfall["Algorithm"] == "Pysindy"]
print(f"Mean time per run (Burgers eq., Pysindy al.): {time.time.mean():.4f} s")

path = 'data_kdv_sindy/'
names = ["0", "1e-5", "3.5e-5", "5.5e-5", "8e-5", "0.0001", "2.26e-4"]
n_df = 3
dfall = melt_count_time(path, names, n_df)
time = dfall[dfall["Algorithm"] == "Pysindy"]
print(f"Mean time per run (KdV     eq., Pysindy al.): {time.time.mean():.4f} s")

