import numpy as np
import pandas as pd
import statsmodels.api as sm
from sklearn.linear_model import LassoCV
from sklearn.preprocessing import StandardScaler

def one_stage_ols(X, Y):
    X_constant = sm.add_constant(X, has_constant='add')
    ols = sm.OLS(Y, X_constant).fit()
    return ols.params[1]  # beta_hat

def two_stage_ols(X, Y, G):
    G_constant = sm.add_constant(G, has_constant='add')
    stage_1 = sm.OLS(X, G_constant).fit()
    X_hat = stage_1.fittedvalues
    X_hat_constant = sm.add_constant(X_hat, has_constant='add')
    stage_2 = sm.OLS(Y, X_hat_constant).fit()
    return stage_2.params[1]  # beta_hat

def two_stage_ols_LASSO(X, Y, G):
    scaler = StandardScaler(with_mean=True, with_std=True)
    G_scaler = scaler.fit_transform(G)
    lasso = LassoCV(cv=5)
    lasso.fit(G_scaler, X)
    X_hat = lasso.predict(G_scaler)
    X_hat_constant = sm.add_constant(X_hat, has_constant='add')
    stage_2 = sm.OLS(Y, X_hat_constant).fit()
    return stage_2.params[1]  # beta_hat

# unique values
# define parameters
n_list = [1000, 3000, 5000, 10000, 20000]
confounding_list = [True, False]
L_list = [5, 10, 15, 30, 50]  # number of SNPs to test

results = []

for n in n_list:
    for L in L_list:
        for confounding in confounding_list:
            fname = f"data/GWAS-MR-{n}-{L}-0.3-{confounding}-7.csv"
            data = pd.read_csv(fname, sep="\t", header=0)
            instrument_cols = [c for c in data.columns if c.startswith('G')]
            Y = data['Y'].values
            X = data['X'].values
            G = data[instrument_cols].values

            results.append({"n": int(n), "L": int(L), "confounding": confounding,
                            "method": "OLS", "beta_hat": one_stage_ols(X, Y)})
            results.append({"n": int(n), "L": int(L), "confounding": confounding,
                            "method": "2SLS", "beta_hat": two_stage_ols(X, Y, G)})
            results.append({"n": int(n), "L": int(L), "confounding": confounding,
                            "method": "2SLS-LASSO", "beta_hat": two_stage_ols_LASSO(X, Y, G)})

df_results = pd.DataFrame(results)
df_results.to_csv("models/baseline.csv", header=True, index=False, sep="\t")