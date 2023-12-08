import matplotlib.pyplot as plt
import matplotlib.colors
import matplotlib as mpl
import seaborn as sns
import numpy as np
from drawbase.read_compile_df import read_compile_mae_df
from drawbase.preprocess_df import melt_count_mae
sns.set(style="whitegrid", color_codes=True)


def plot_mae(path, names, core_values, core_colors, decimals=4, n_df=2, log_transform=False):
    count_ls = read_compile_mae_df(path, names, decimals=decimals, n_df=n_df)
    dfall = melt_count_mae(count_ls)

    idxs, color_keys = [], []
    cmap, norm, color_dict, mae = _create_cmap(core_values, core_colors, count_ls, log_transform)
    for i, g in enumerate(dfall.groupby("variable")):
        ax = sns.barplot(data=g[1],
                         x="index",
                         y="vcs",
                         hue="Name",
                         zorder=-i,
                         edgecolor="k")
        ax.set_axisbelow(True)
        color_keys += [g[0]] * len(g[1])
        idxs += list(g[1].Name.values)

    ax.legend_.remove()
    plt.grid(False)
    # hatches = ['-', '+', 'x', '\\', '*', 'o']
    for j, thisbar in enumerate(ax.patches):
        thisbar._facecolor = tuple(color_dict.get(color_keys[j]))
        if idxs[j] == "Modified":
            thisbar.set_hatch("\\")
        elif idxs[j] == "Pysindy":
            thisbar.set_hatch("-")

    keys_sm = list(color_dict.keys())[:len(color_dict) - 1] # listed
    vals_sm = list(color_dict.values())[:len(color_dict) - 1] # colors
    cmap_sm = mpl.colors.ListedColormap(vals_sm)
    bounds = keys_sm.copy()
    bounds.append(keys_sm[-1] + keys_sm[-1] - keys_sm[-2])
    norm_sm = mpl.colors.BoundaryNorm(bounds, cmap_sm.N)

    sm = plt.cm.ScalarMappable(cmap=cmap_sm, norm=norm_sm)
    sm.set_array([])
    cbar = plt.colorbar(sm, ticks=keys_sm, boundaries=bounds)

    locations = []
    for i in range(0, len(bounds) - 1):
        locations.append((bounds[i] + bounds[i + 1]) / 2)
    cbar.set_ticks(locations)
    cbar.ax.set_yticklabels(keys_sm, fontsize=24)

    n = []
    for i in range(len(dfall.Name.unique())):
        if i == 2:
            n.append(ax.bar(0, 0, color="#BFBFBF", hatch="-", edgecolor="k"))
        elif i == 1:
            n.append(ax.bar(0, 0, color="#BFBFBF", hatch="\\", edgecolor="k"))
        else:
            n.append(ax.bar(0, 0, color="#BFBFBF", edgecolor="k"))

    plt.legend(n, dfall.Name.unique(), loc=[0.83, 0.84])
    plt.setp(ax.get_legend().get_texts(), fontsize='24')
    ax.set_ylabel("No. of runs", fontsize=24)
    ax.set_xlabel("Magnitude", fontsize=24)
    plt.xticks(fontsize=24, rotation=0)
    plt.yticks(fontsize=24)
    plt.show()


def _create_cmap(core_values, core_colors, count_ls, log_transform, plot_map=False):
    listed = list(count_ls[0].columns)
    listed.pop()
    listed.pop()
    if log_transform:
        cat_lg = np.log(listed)
        core_values_lg = np.log(core_values)
        norm = plt.Normalize(min(cat_lg), max(cat_lg))
        tuples = list(zip(map(norm, core_values_lg), core_colors))
        cmap = matplotlib.colors.LinearSegmentedColormap.from_list("", tuples)
        colors = cmap(norm(cat_lg))
        col_dict = dict(zip(listed, colors))
    else:
        norm = plt.Normalize(min(listed), max(listed))
        tuples = list(zip(map(norm, core_values), core_colors))
        cmap = matplotlib.colors.LinearSegmentedColormap.from_list("", tuples)
        colors = cmap(norm(listed))
        col_dict = dict(zip(listed, colors))
    col_dict["n/a"] = np.array([191. / 255, 191. / 255, 191. / 255, 1.0])

    if plot_map:
        x = np.linspace(0, 6, len(listed))
        y = np.zeros((len(listed),))
        plt.scatter(x, y, c=listed, cmap=cmap, norm=norm, s=90)
        plt.show()
    return cmap, norm, col_dict, listed
