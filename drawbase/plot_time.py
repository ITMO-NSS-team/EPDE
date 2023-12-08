import seaborn as sns
import matplotlib.pyplot as plt
from drawbase.preprocess_df import melt_count_time
sns.set(style="whitegrid", color_codes=True)


def plot_time(path, names, n_df):
    dfall = melt_count_time(path, names, n_df)
    lbl_y = "s"
    if dfall.time.max() > 150:
        lbl_y = "m"

    ax = sns.boxplot(x=dfall['Magnitude'],
                y=dfall['time'],
                hue=dfall['Algorithm'],
                showfliers=False)
    plt.xticks(fontsize=24, rotation=0)
    plt.yticks(fontsize=24)
    plt.setp(ax.get_legend().get_texts(), fontsize='24')
    plt.setp(ax.get_legend().get_title(), fontsize='24')
    ax.set_ylabel(f"Time, {lbl_y}", fontsize=24)
    ax.set_xlabel("Magnitude", fontsize=24)
    plt.show()


def plot_sindy(path, names, n_df):
    dfall = melt_count_time(path, names, n_df)

    ratio_k = 4
    fig, (ax1, ax2) = plt.subplots(figsize=(16, 8), ncols=1, nrows=2, sharex=True,
                                   gridspec_kw={'hspace': 0.07, 'height_ratios': [ratio_k, 1]})
    sns.boxplot(x=dfall['Magnitude'],
                y=dfall['time'],
                hue=dfall['Algorithm'],
                orient="v",
                showfliers=False,
                ax=ax1)
    sns.boxplot(x=dfall['Magnitude'],
                y=dfall['time'],
                hue=dfall['Algorithm'],
                orient="v",
                showfliers=False,
                ax=ax2)
    ax1.set_ylim(24.2, 31.7)
    ax2.set_ylim(0.022, 0.028)

    d = .005
    kwargs = dict(transform=ax1.transAxes, color="k", clip_on=False)
    ax1.plot((-d, +d), (-d, +d), **kwargs)
    ax1.plot((1 - d, 1 + d), (-d, +d), **kwargs)
    kwargs.update(transform=ax2.transAxes)
    ax2.plot((-d, +d), (1 - ratio_k * d, 1 + ratio_k * d), **kwargs)
    ax2.plot((1 - d, 1 + d), (1 - ratio_k * d, 1 + ratio_k * d), **kwargs)

    ax1.tick_params(axis='both', which='major', labelsize=24)
    ax1.tick_params(axis='both', which='minor', labelsize=24)

    ax2.tick_params(axis='both', which='major', labelsize=24)
    ax2.tick_params(axis='both', which='minor', labelsize=24)
    ax2.tick_params(axis='x', labelrotation=25, bottom=True)

    ax1.grid()
    ax2.grid()
    fig.text(0.065, 0.5, "time, s", va="center", rotation="vertical", fontsize=24)

    ax1.xaxis.tick_bottom()
    plt.subplots_adjust(bottom=0.189, top=1.)
    plt.show()
