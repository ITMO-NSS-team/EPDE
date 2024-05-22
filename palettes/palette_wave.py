import matplotlib.colors
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import os
from pathlib import Path
from drawbase.read_compile_df import _round_values, read_csv

sns.set(style="whitegrid", color_codes=True)

path = 'data_wave/'
path = str(os.path.join(Path().absolute().parent, path))
# names = ["0", "2e-5", "2.5e-5", "3e-5", "3.2e-5", "3.47e-5"]
names = ["0", "8.675e-6", "1.735e-5", "2.6025e-5", "3.47e-5"]
n_df = 2
decimals = 3

df_ls = read_csv(path, names, n_df)
categories, df_lsr = _round_values(df_ls, decimals=decimals)
categories = categories[np.argsort(categories)]
categories = categories[:len(categories)-1]

# categories = [0.0087, 0.0088, 0.0089, 0.009,  0.0091, 0.0092, 0.0093, 0.0445, 0.045, 0.0457]
core_values = [0.009, 0.044, 0.045]
core_colors = ["#385623", "#71AE48", "#A8D08D"]
# core_values = [0.0087, 0.0093, 0.0445, 0.0452]
# core_colors = ["#538135", "#89BF65", "#C5E0B3", "#E2EFD9"]

norm=plt.Normalize(min(categories),max(categories))
tuples = list(zip(map(norm,core_values), core_colors))
cmap = matplotlib.colors.LinearSegmentedColormap.from_list("", tuples)

x = np.linspace(0, 6, len(categories))
y = np.zeros((len(categories), ))
colors = cmap(norm(categories))

plt.scatter(x,y,c=categories, cmap=cmap, norm=norm, s=1000)
plt.show()
