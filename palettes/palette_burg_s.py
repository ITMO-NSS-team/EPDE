import matplotlib.colors
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from drawbase.read_compile_df import _round_values, read_csv

import os
from pathlib import Path
sns.set(style="whitegrid", color_codes=True)

path = 'data_burg_sindy/'
path = str(os.path.join(Path().absolute().parent, path))
# names = ["0", "0.001", "0.005", "0.01", "0.02", "0.03"]
names = ["0", "0.0075",  "0.015",  "0.0225", "0.03"]
n_df = 3
decimals = [3, 1]

df_ls = read_csv(path, names, n_df)
categories, df_lsr = _round_values(df_ls, decimals=decimals)
categories = categories[np.argsort(categories)]
categories = categories[:len(categories)-1]

# core_values = [8.0e-05, 2.0e-03, 6.0e-03, 1.3e-02, 2.1e-02, 1.0e-01, 2.0e-01]
# core_colors = ["#385623", "#43682A", "#538135", "#71AE48", "#C5E0B3", "#E2EFD9", "#F4F9F1"]
core_values = [8.0e-05, 2.0e-03, 8.0e-03, 2.0e-02, 2.0e-01]
core_colors = ["#385623", "#538135", "#669D41", "#C5E0B3", "#FDFEFC"]

norm=plt.Normalize(min(categories),max(categories))
tuples = list(zip(map(norm,core_values), core_colors))
cmap = matplotlib.colors.LinearSegmentedColormap.from_list("", tuples)

x = np.linspace(0, 6, len(categories))
y = np.zeros((len(categories), ))
colors = cmap(norm(categories))

plt.scatter(x,y,c=categories, cmap=cmap, norm=norm, s=1000)
plt.show()
