import pandas as pd
import matplotlib.pyplot as plt
import re
from pysindy_calc_mae import calc_difference

# line = '-0.991591924070889 x0_111 + -5.966916815057163 x0x0_1 + 0.88888888888888 x0'
line = '-0.99159192407088903032 x0_111 + -5.96691681505716342571 x0x0_1'

mae1 = abs(float(-0.99159192407088903032) + 1) + (float(-5.96691681505716342571) + 6)
mae2 = mae1 / 3

true_coef = [-1, -6]
true_names = ["x0_111", "x0x0_1"]
mae, found_flag = calc_difference(line, true_coef, true_names)
idx = true_names.index("x0")

# mae = 0
# term_found = 0
# for i in range(len(names)):
#     if names[i] in true_names:
#         term_found += 1
#         idx = true_names.index(names[i])
#         mae += abs(coeffs[i] - true_coef[idx])
#     else:
#         mae += abs(coeffs[i])
#     mae /= (len(names) + 1)

print()

# magnitudes = [1. * 1e-5, 1.4 * 1e-5, 1.8 * 1e-5, 2.2 * 1e-5, 2.6 * 1e-5, 3. * 1e-5]
# df2 = pd.read_csv("data_burg/dfs1.4e-05.csv", index_col="Unnamed: 0")
# df3 = pd.read_csv("data_burg/dfs1.8e-05.csv", index_col="Unnamed: 0")
# df1 = pd.read_csv("data_burg/dfs1e-05.csv", index_col="Unnamed: 0")
# df4 = pd.read_csv("data_burg/dfs2.2000000000000003e-05.csv", index_col="Unnamed: 0")
# df5 = pd.read_csv("data_burg/dfs2.6000000000000002e-05.csv", index_col="Unnamed: 0")
# df6 = pd.read_csv("data_burg/dfs3.0000000000000004e-05.csv", index_col="Unnamed: 0")
# df_ls = [df1, df2, df3, df4, df5, df6]
#
# draw_mae = [df1.MAE.mean(), df2.MAE.mean(), df3.MAE.mean(), df4.MAE.mean(), df5.MAE.mean(), df6.MAE.mean()]
# draw_num = [sum(df.number_found_eq == 0) for df in df_ls]
# draw_time = [sum(df.time)/len(df.time) for df in df_ls]
# plt.plot(magnitudes, draw_num, linewidth=2, markersize=9, marker='o')
# plt.ylabel("No. runs with not found eq.")
# plt.xlabel("Magnitude value")
# plt.grid()
# plt.show()
#
# plt.plot(magnitudes, draw_time, linewidth=2, markersize=9, marker='o')
# plt.ylabel("Time, s.")
# plt.xlabel("Magnitude value")
# plt.grid()
# plt.show()
#
# plt.plot(magnitudes, draw_mae, linewidth=2, markersize=9, marker='o')
# plt.ylabel("Average MAE")
# plt.xlabel("Magnitude value")
# plt.grid()
# plt.show()
