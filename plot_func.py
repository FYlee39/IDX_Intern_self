import matplotlib.pyplot as plt
import numpy as np

def plot_hexbin(df,
                target_col,
                method="mean",
                lat_col="Latitude",
                lon_col="Longitude",
                gridsize=60):
    if method not in {"mean", "median"}:
        raise ValueError("method must be either 'mean' or 'median'")

    sub = df[[lat_col, lon_col, target_col]].dropna()

    plt.figure()
    plt.hexbin(
        sub[lon_col], sub[lat_col],
        C=sub[target_col],
        reduce_C_function=np.mean if method == "mean" else np.median,
        gridsize=gridsize,
        mincnt=20
    )
    plt.colorbar(label=f"{target_col} {method} in cell")
    plt.xlabel("Longitude")
    plt.ylabel("Latitude")
    plt.title(f"{target_col} by Location ({method} hexbin)")
    plt.show()
