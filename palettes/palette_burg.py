import matplotlib.colors
import matplotlib.pyplot as plt
from drawbase.read_compile_df import _round_values, read_csv
import seaborn as sns
import numpy as np
import os
from pathlib import Path
sns.set(style="whitegrid", color_codes=True)
plt.rc('axes', axisbelow=True)


path = 'data_burg/'
path = str(os.path.join(Path().absolute().parent, path))
names = ["0", "1e-5", "1.5e-5", "2e-5", "2.5e-5", "3e-5", "3.67e-5"]
df_ls = read_csv(path, names, 2)
categories, df_lsr = _round_values(df_ls, decimals=4)
categories = categories[np.argsort(categories)]
categories = categories[:len(categories)-1]

core_values = [0.0006, 0.0008, 0.001, 0.0014, 0.0132, 0.0137, 0.014, 0.0146]
core_colors = ["#385623", "#538135", "#669D41", "#71AE48", "#89BF65", "#A8D08D", "#C5E0B3", "#E2EFD9"]

norm=plt.Normalize(min(categories),max(categories))
tuples = list(zip(map(norm,core_values), core_colors))
cmap = matplotlib.colors.LinearSegmentedColormap.from_list("", tuples)

x = np.linspace(0, 6, len(categories))
y = np.zeros((len(categories), ))
colors = cmap(norm(categories))

plt.scatter(x,y,c=categories, cmap=cmap, norm=norm, s=1000)
plt.show()


# df0o = pd.read_csv(f'{path}dfo.csv', index_col="Unnamed: 0")
# df0s = pd.read_csv(f'{path}dfs.csv', index_col="Unnamed: 0")
# df1o = pd.read_csv(f'{path}dfo1e-05.csv', index_col="Unnamed: 0")
# df1s = pd.read_csv(f'{path}dfs1e-05.csv', index_col="Unnamed: 0")
# df2o = pd.read_csv(f'{path}dfo1.5000000000000002e-05.csv', index_col="Unnamed: 0")
# df2s = pd.read_csv(f'{path}dfs1.5000000000000002e-05.csv', index_col="Unnamed: 0")
# df3o = pd.read_csv(f'{path}dfo2e-05.csv', index_col="Unnamed: 0")
# df3s = pd.read_csv(f'{path}dfs2e-05.csv', index_col="Unnamed: 0")
# df4o = pd.read_csv(f'{path}dfo2.5e-05.csv', index_col="Unnamed: 0")
# df4s = pd.read_csv(f'{path}dfs2.5e-05.csv', index_col="Unnamed: 0")
# df5o = pd.read_csv(f'{path}dfo3.0000000000000004e-05.csv', index_col="Unnamed: 0")
# df5s = pd.read_csv(f'{path}dfs3.0000000000000004e-05.csv', index_col="Unnamed: 0")
# df6o = pd.read_csv(f'{path}dfo3.6700000000000004e-05.csv', index_col="Unnamed: 0")
# df6s = pd.read_csv(f'{path}dfs3.6700000000000004e-05.csv', index_col="Unnamed: 0")
#
# dfo_ls = [df0o, df1o, df2o, df3o, df4o, df5o, df6o]
# dfs_ls = [df0s, df1s, df2s, df3s, df4s, df5s, df6s]


# count_o, count_s = make_input_df(names, dfo_ls, dfs_ls)
# listed = list(count_o.columns)
# listed.pop()


# norm=plt.Normalize(min(listed),max(listed))
# tuples = list(zip(map(norm,core_values), core_colors))
# cmap = matplotlib.colors.LinearSegmentedColormap.from_list("", tuples)
#
# colors = cmap(norm(listed))
# colors = np.append(colors, [[191./255, 191./255, 191./255, 1.0]], 0)
#
# mini, maxi = min(listed), max(listed)
# x = np.linspace(0, 6, 23)
# y = np.zeros((23, ))
#
# plt.scatter(x,y,c=listed, cmap=cmap, norm=norm, s=90)
# plt.show()