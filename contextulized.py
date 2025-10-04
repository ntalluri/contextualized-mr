import pandas as pd
import statsmodels.api as sm
from sklearn.linear_model import LassoCV
from sklearn.preprocessing import StandardScaler
from contextualized.easy import ContextualizedRegressor

def one_stage_contextualized(X, Y, C):
    # model = ContextualizedRegressor(encoder_type="mlp")

    model = ContextualizedRegressor(
        encoder_type="mlp",
        width=8,
        depth=1,
        num_archetypes=0
    )

    model.fit(C, X.reshape(-1,1), Y) # needs to be a 2d array
    
    betas, mus = model.predict_params(C)
    return model, betas, mus


def two_stage_contextualized(X, Y, G, C):
    # Stage 1: X ~ G (plain OLS)
    G_const = sm.add_constant(G, has_constant='add')
    stage1 = sm.OLS(X, G_const).fit()
    X_hat = stage1.fittedvalues.reshape(-1,1) # needs to be a 2d array

    # Stage 2: contextualized regression of Y ~ X_hat, context
    # model = ContextualizedRegressor(encoder_type="mlp")
    model = ContextualizedRegressor(
        encoder_type="mlp",
        width=8,
        depth=1,
        num_archetypes=0
    )
    model.fit(C, X_hat, Y)
    
    
    betas, mus = model.predict_params(C)
    return model, betas, mus


def two_stage_contextualized_LASSO(X, Y, G, C):
    # Stage 1: X ~ G (LASSO)
    scaler = StandardScaler()
    G_s = scaler.fit_transform(G)
    lasso = LassoCV(cv=5).fit(G_s, X)
    X_hat = lasso.predict(G_s).reshape(-1,1) # needs to be a 2d array

    # Stage 2: contextualized regression of Y ~ X_hat, context
    # model = ContextualizedRegressor(encoder_type="mlp")
    model = ContextualizedRegressor(
        encoder_type="mlp",
        width=8,
        depth=1,
        num_archetypes=0
    )
    model.fit(C, X_hat, Y)
    
    betas, mus = model.predict_params(C)
    return model, betas, mus

n_list = ["1000", "3000", "5000", "10000", "20000"]
confounding_list = ["True", "False"]
L_list = ["5", "10", "15", "30", "50"]

# n_list = ["1000"]
# confounding_list = ["True"]
# L_list = ["5"]

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
            
            cols = ["age","sex","bmi","smoker","height","weight","bp_sys","cholesterol","glucose","sleep"]
            C_raw = data[cols].values
            C_mean = C_raw.mean(axis=0)
            C_std  = C_raw.std(axis=0, ddof=0)
            C = (C_raw - C_mean) / C_std


            model, betas, mus = one_stage_contextualized(X, Y, C)
            results.append({
                "n": int(n),
                "L": int(L),
                "confounding": confounding,
                "method": "OLS",
                "beta_hat_avg": betas.mean(),
                "beta_hat_all": betas.squeeze().tolist()
            })

            _, betas, _ = two_stage_contextualized(X, Y, G, C)
            results.append({
                "n": int(n),
                "L": int(L),
                "confounding": confounding,
                "method": "2SLS",
                "beta_hat_avg": betas.mean(),
                "beta_hat_all": betas.squeeze().tolist()
            })
            
            _, betas, _ = two_stage_contextualized_LASSO(X, Y, G, C)
            results.append({
                "n": int(n),
                "L": int(L),
                "confounding": confounding,
                "method": "2SLS-LASSO",
                "beta_hat_avg": betas.mean(),
                "beta_hat_all": betas.squeeze().tolist()
            })


df_results = pd.DataFrame(results)
print(df_results)
df_results.to_csv("models/contextulized-small-mlp.csv", header=True, index=False, sep="\t")
