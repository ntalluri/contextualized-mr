import pandas as pd
import numpy as np
import statsmodels.api as sm
import ast

beta_true = 0.3

df_results = pd.read_csv("models/contextulized-as-is.csv", header=0, sep="\t")
df_results["method"] = df_results["method"].replace({
    "OLS": "Contextulized-MLP-OLS",
    "2SLS": "Contextulized-MLP-2SLS",
    "2SLS-LASSO": "Contextulized-MLP-2SLS-LASSO"
})

all_mses = []
for idx, row in df_results.iterrows():
    betas = np.array(ast.literal_eval(row["beta_hat_all"]))
    squared_errors = (betas - beta_true) ** 2
    mse = squared_errors.mean()
    all_mses.append(mse)

df_results["mse"] = all_mses
df_results["label"] = df_results["n"].astype(str) + "-" + df_results["L"].astype(str) + "-" + df_results["confounding"].astype(str)
df_results.to_csv("outputs/contextulized-as-is.csv", header=True, index=False, sep="\t")


df_results = pd.read_csv("models/contextulized-small-mlp.csv", header=0, sep="\t")
df_results["method"] = df_results["method"].replace({
    "OLS": "Contextulized-SmallMLP-OLS",
    "2SLS": "Contextulized-SmallMLP-2SLS",
    "2SLS-LASSO": "Contextulized-SmallMLP-2SLS-LASSO"
})

all_mses = []
for idx, row in df_results.iterrows():
    betas = np.array(ast.literal_eval(row["beta_hat_all"]))
    squared_errors = (betas - beta_true) ** 2
    mse = squared_errors.mean()
    all_mses.append(mse)

df_results["mse"] = all_mses
df_results["label"] = df_results["n"].astype(str) + "-" + df_results["L"].astype(str) + "-" + df_results["confounding"].astype(str)
df_results.to_csv("outputs/contextulized-small-mlp-mse.csv", header=True, index=False, sep="\t")


df_results = pd.read_csv("models/baseline.csv", header=0, sep="\t")

all_mses = []
for idx, row in df_results.iterrows():
    squared_error = (row["beta_hat"] - beta_true) ** 2
    all_mses.append(squared_error)

df_results["mse"] = all_mses
df_results["label"] = df_results["n"].astype(str) + "-" + df_results["L"].astype(str) + "-" + df_results["confounding"].astype(str)
df_results.to_csv("outputs/baseline-mse.csv", header=True, index=False, sep="\t")
