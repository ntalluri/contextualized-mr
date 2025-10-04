import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import statsmodels.api as sm
from sklearn.linear_model import LassoCV
from sklearn.preprocessing import StandardScaler
import seaborn as sns

n_list = [1000, 3000, 5000, 10000, 20000]
confounding_list = [True, False]

df_results = pd.read_csv("models/contextulized-as-is.csv", header = 0, sep="\t")

for n in n_list:
    subset = df_results[df_results["n"] == n]
    

    fig, axes = plt.subplots(1, 2, figsize=(12, 5), sharey=True)

    for i, conf in enumerate(confounding_list):
        sub = subset[subset["confounding"] == conf]
        
        sns.pointplot(
            data=sub,
            x="L", y="beta_hat_avg", hue="method",
            dodge=True, markers=["o", "s", "v"], ax=axes[i]
        )
        axes[i].axhline(0.3, color="red", linestyle="--", label="True β")
        axes[i].set_title(f"Contextulized Model Default MLP: n = {n}, Confounding = {conf}")
        axes[i].set_xlabel("Number of instruments (L)")
        if i == 0:
            axes[i].set_ylabel("β̂")
        else:
            axes[i].set_ylabel("")

    plt.tight_layout()
    plt.savefig(f"plots/plot-beta-n{n}-by-L-context-as-is.png")
    plt.close()


n_list = [1000, 3000, 5000, 10000, 20000]
confounding_list = [True, False]

df_results = pd.read_csv("models/contextulized-small-mlp.csv", header = 0, sep="\t")

for n in n_list:
    subset = df_results[df_results["n"] == n]
    

    fig, axes = plt.subplots(1, 2, figsize=(12, 5), sharey=True)

    for i, conf in enumerate(confounding_list):
        sub = subset[subset["confounding"] == conf]
        
        sns.pointplot(
            data=sub,
            x="L", y="beta_hat_avg", hue="method",
            dodge=True, markers=["o", "s", "v"], ax=axes[i]
        )
        axes[i].axhline(0.3, color="red", linestyle="--", label="True β")
        axes[i].set_title(f"Contextulized Model Small MLP: n = {n}, Confounding = {conf}")
        axes[i].set_xlabel("Number of instruments (L)")
        if i == 0:
            axes[i].set_ylabel("β̂")
        else:
            axes[i].set_ylabel("")

    plt.tight_layout()
    plt.savefig(f"plots/plot-beta-n{n}-by-L-context-small-mlp.png")
    plt.close()

# import seaborn as sns
# import pandas as pd
# import numpy as np
# import ast

# df_results = pd.read_csv("plots_test/plot-n-L-contextulized.csv", header = 0, sep="\t")
# betas = df_results["beta_hat_all"][0]
# betas = np.array(ast.literal_eval(betas)).squeeze()
# print(type(betas))

# data = pd.read_csv("data/GWAS-MR-1000-5-0.3-True-7.csv", header = 0, sep="\t")
# cols = ["age","sex","bmi","smoker","height","weight",
#         "bp_sys","cholesterol","glucose","sleep"]
# C_raw = data[cols].values
# sex = C_raw[:, 1].squeeze()
# print(type(sex))

# sns.boxplot(x=sex, y=betas)  # col 1 = sex
# plt.xlabel("Sex (0=female, 1=male)")
# plt.ylabel("Estimated causal effect (β)")
# plt.title("Contextualized β by Sex")
# plt.show()



# bmi = C_raw[:, 2].squeeze()
# print(type(bmi))

# plt.scatter(bmi, betas, alpha=0.5)
# plt.xlabel("BMI")
# plt.ylabel("Estimated causal effect (β)")
# plt.title("Contextualized β by Sex")
# plt.show()

# age = C_raw[:, 0].squeeze()
# print(type(age))

# plt.scatter(age, betas, alpha=0.5)
# plt.xlabel("Age")
# plt.ylabel("Estimated causal effect (β)")
# plt.title("Contextualized β by Sex")
# plt.show()