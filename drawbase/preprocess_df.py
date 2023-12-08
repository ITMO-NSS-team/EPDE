import pandas as pd
import numpy as np
from drawbase.read_compile_df import read_csv


def melt_count_mae(count_ls):

    count_ls[0]["Name"] = "Classical"
    count_ls[1]["Name"] = "Modified"
    if len(count_ls) == 3:
        count_ls[2]["Name"] = "Pysindy"

    dfall = pd.concat([pd.melt(i.reset_index(),
                               id_vars=["Name", "index"])
                       for i in count_ls],
                      ignore_index=True)
    dfall.set_index(["Name", "index", "variable"], inplace=True)
    dfall["vcs"] = dfall.groupby(level=["Name", "index"]).cumsum()
    dfall.reset_index(inplace=True)

    for i in range(len(dfall)):
        if np.isnan(dfall.loc[i, "variable"]):
            dfall.loc[i, "variable"] = "n/a"
    return dfall


def melt_count_time(path, names, n_df):
    df_lsls = read_csv(path, names, n_df)
    for df in df_lsls[0]:
        df["Algorithm"] = "Classical"
    for df in df_lsls[1]:
        df["Algorithm"] = "Modified"
    if n_df == 3:
        for df in df_lsls[2]:
            df["Algorithm"] = "Pysindy"

    subdf_lsls = []
    for i in range(len(df_lsls)):
        for j in range(len(names)):
            temp_df = df_lsls[i][j][["time", "Algorithm"]]
            temp_df["Magnitude"] = names[j]
            subdf_lsls.append(temp_df)
    dfall = pd.concat(subdf_lsls, ignore_index=True)
    return dfall