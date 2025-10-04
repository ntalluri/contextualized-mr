import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import statsmodels.api as sm
from sklearn.linear_model import LassoCV
from sklearn.preprocessing import StandardScaler
import seaborn as sns

# TODO add code to make the plots folder
# and set a folder var to use for input data and output data

df_results = pd.read_csv("models/baseline.csv", header=0, sep="\t")
n_list = [1000, 3000, 5000, 10000, 20000]
confounding_list = [True, False] 

for n in n_list:
    subset = df_results[df_results["n"] == n]

    fig, axes = plt.subplots(1, 2, figsize=(12, 5), sharey=True)

    for i, conf in enumerate(confounding_list):
        sub = subset[subset["confounding"] == conf]

        sns.pointplot(
            data=sub,
            x="L", y="beta_hat", hue="method",
            dodge=True, markers=["o", "s", "v"], ax=axes[i]
        )
        axes[i].axhline(0.3, color="red", linestyle="--", label="True β")
        axes[i].set_title(f"Baseline Models: n = {n}, Confounding = {conf}")
        axes[i].set_xlabel("Number of instruments (L)")
        if i == 0:
            axes[i].set_ylabel("β̂")
        else:
            axes[i].set_ylabel("")

    plt.tight_layout()
    plt.savefig(f"plots/plot-beta-n{n}-by-L-baseline.png")
    plt.close()
