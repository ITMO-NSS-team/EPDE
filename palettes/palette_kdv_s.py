import matplotlib.colors
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from drawbase.read_compile_df import _round_values, read_csv
import os
from pathlib import Path
sns.set(style="whitegrid", color_codes=True)


path = 'data_kdv_sindy/'
path = str(os.path.join(Path().absolute().parent, path))
names = ["0", "1e-5", "3.5e-5", "5.5e-5", "8e-5", "0.0001", "2.26e-4"]
n_df = 3
decimals = [3, 1]

df_ls = read_csv(path, names, n_df)
categories, df_lsr = _round_values(df_ls, decimals=decimals)
categories = categories[np.argsort(categories)]
categories = categories[:len(categories)-1]

core_values = [3.0e-03, 8.0e-03, 1.0e-02, 1.4e-02, 1.9e-02, 2.1e-02, 4.4e-02, 1.6, 3.2]
core_colors = ["#385623", "#43682A", "#5A8B39", "#669D41", "#89BF65", "#A8D08D", "#C5E0B3", "#E2EFD9", "#FDFEFC"]

categories_log = np.log(categories)
core_values_log = np.log(core_values)
norm=plt.Normalize(min(categories_log),max(categories_log))
tuples = list(zip(map(norm,core_values_log), core_colors))
cmap = matplotlib.colors.LinearSegmentedColormap.from_list("", tuples)

listed1 = np.log(categories)
listed = cmap(norm(categories_log))
x = np.linspace(0, 6, len(categories))
y = np.zeros((len(categories), ))
colors = cmap(norm(categories_log))

plt.scatter(x,y,c=categories_log, cmap=cmap, norm=norm, s=1000)
plt.show()
