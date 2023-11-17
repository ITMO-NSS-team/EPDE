import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib import rcParams
rcParams.update({'figure.autolayout': True})


path = 'data_burg/'
# df0 = pd.read_csv(f'{path}df0.csv')
df1 = pd.read_csv(f'{path}dfs.csv', index_col="Unnamed: 0")
# df1 = pd.read_csv(f'{path}dfs.csv')
# df2 = pd.read_csv(f'{path}df2.csv')
# df3 = pd.read_csv(f'{path}df3.csv')
# df4 = pd.read_csv(f'{path}df4.csv')
df_ls = [df1]


title = "burg_mae"
feature = 'MAE'
columns = [f'symnet algorithm',]
new_df = pd.DataFrame(columns=columns)
for i in range(len(columns)):
    new_df[columns[i]] = df_ls[i][feature]

fig, ax = plt.subplots(figsize=(16, 8))
sns.boxplot(data=new_df[columns], orient="v", ax=ax, showfliers=False)
plt.grid()
plt.ylabel("MAE, s", fontsize=24)
plt.autoscale()
plt.xticks(fontsize=24, rotation=25)
plt.yticks(fontsize=24)
# plt.savefig(f'{data}time_boxplot.png')
plt.show()






# path = 'data_burg/'
# # df0 = pd.read_csv(f'{path}df0.csv')
# df1 = pd.read_csv(f'{path}dfs.csv', index_col="Unnamed: 0")
# # df1 = pd.read_csv(f'{path}dfs.csv')
# # df2 = pd.read_csv(f'{path}df2.csv')
# # df3 = pd.read_csv(f'{path}df3.csv')
# # df4 = pd.read_csv(f'{path}df4.csv')
# df_ls = [df1]
#
#
# title = "burg_mae"
# feature = 'MAE'
# columns = [f'Basic algorithm', f'Fixed initial distr.', f'Altered initial distr.',
#            f'Highly altered distr.', f'Uniform initial distr.']
# new_df = pd.DataFrame(columns=columns)
# for i in range(len(columns)):
#     col_values = df_ls[i][feature]
#     col_values = np.round(col_values, 4)
#     new_df[columns[i]] = col_values
#
# matrix = new_df.values
# new_df_text = pd.DataFrame(columns=columns)
# ls_column = []
# for i in range(len(columns)):
#     ls_column = []
#     for elem in matrix[:, i]:
#         ls_column.append(str(elem))
#     new_df_text[columns[i]] = ls_column
#
# new_df.to_csv(f'{path}{title}.csv', index=False)
#
#
# feature = 'time'
# columns = [f'basic_algorithm', f'fixed_initial_distr', f'biased_distr',
#            f'highly_biased_distr', f'uniform_distr']
# new_df = pd.DataFrame(columns=columns)
# for i in range(len(columns)):
#     new_df[columns[i]] = df_ls[i][feature]
#
# fig, ax = plt.subplots(figsize=(16, 8))
# sns.boxplot(data=new_df[columns], orient="v", ax=ax, showfliers=True)
# plt.grid()
# plt.ylabel("time, s", fontsize=24)
# plt.autoscale()
# plt.xticks(fontsize=24, rotation=25)
# plt.yticks(fontsize=24)
# # plt.savefig(f'{data}time_boxplot.png')
# plt.show()
