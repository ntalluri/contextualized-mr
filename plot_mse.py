import matplotlib.pyplot as plt
import pandas as pd
import statsmodels.api as sm
import seaborn as sns

contextulized_as_is = pd.read_csv("outputs/contextulized-as-is.csv", header=0, sep="\t")
plt.figure(figsize=(15, 6))
sns.barplot(data=contextulized_as_is, x="label", y="mse", hue="method")
plt.title("MSE by Method and Condition (for contextulized-mlp)")
plt.xlabel("Condition (n-L-context)")
plt.ylabel("MSE")
plt.xticks(rotation=90)
plt.legend(
        bbox_to_anchor=(1.05, 1),
        loc="upper left",
        borderaxespad=0.
    )
plt.tight_layout()
plt.savefig(f"plots/contextulized-mlp.png")
plt.close()

contextulized_small_mlp = pd.read_csv("outputs/contextulized-small-mlp-mse.csv", header=0, sep="\t")
plt.figure(figsize=(15, 6))
sns.barplot(data=contextulized_small_mlp, x="label", y="mse", hue="method")
plt.title("MSE by Method and Condition (for contextulized-small-mlp)")
plt.xlabel("Condition (n-L-context)")
plt.ylabel("MSE")
plt.xticks(rotation=90)
plt.legend(
        bbox_to_anchor=(1.05, 1),
        loc="upper left",
        borderaxespad=0.
    )
plt.tight_layout()
plt.savefig(f"plots/contextulized-small-mlp.png")
plt.close()

baseline = pd.read_csv("outputs/baseline-mse.csv", header=0, sep="\t")
plt.figure(figsize=(15, 6))
sns.barplot(data=baseline, x="label", y="mse", hue="method")
plt.title("MSE by Method and Condition (for baseline)")
plt.xlabel("Condition (n-L-context)")
plt.ylabel("MSE")
plt.xticks(rotation=90)
plt.legend(
        bbox_to_anchor=(1.05, 1),
        loc="upper left",
        borderaxespad=0.
    )
plt.tight_layout()
plt.savefig(f"plots/baseline.png")
plt.close()


cols = ["n", "label", "method", "mse"]
combined = pd.concat([contextulized_as_is[cols], contextulized_small_mlp[cols], baseline[cols]], ignore_index=True)

# plt.figure(figsize=(15, 6))
# sns.barplot(data=combined, x="label", y="mse", hue="method")
# plt.title("MSE by Method and Condition")
# plt.xlabel("Condition (n-L-context)")
# plt.ylabel("MSE")
# plt.xticks(rotation=90)
# plt.legend(
#         bbox_to_anchor=(1.05, 1),
#         loc="upper left",
#         borderaxespad=0.
#     )
# plt.tight_layout()
# plt.savefig(f"plots/all.png")
# plt.close()

for n_val in sorted(combined["n"].unique()):
    subset = combined[combined["n"] == n_val]

    plt.figure(figsize=(15, 6))
    sns.barplot(data=subset, x="label", y="mse", hue="method")
    plt.title(f"MSE by Method and Condition (n={n_val})")
    plt.xlabel("Condition (n-L-context)")
    plt.ylabel("MSE")
    plt.xticks(rotation=90)
    plt.legend(
        bbox_to_anchor=(1.05, 1),
        loc="upper left",
        borderaxespad=0.
    )
    plt.tight_layout()

    # Find lowest mse per label
    best = subset.loc[subset.groupby("label")["mse"].idxmin()]

    # Annotate
    for i, row in best.iterrows():
        x = list(subset["label"].unique()).index(row["label"])
        plt.text(
            x, row["mse"] + 0.01,
            f"best: {row['method']}", 
            ha="center", va="bottom", fontsize=8, rotation=90
        )
    

    plt.savefig(f"plots/mse-n{n_val}.png")
    plt.close()

def get_first_stage(method: str):
    method = method.lower()
    if method == "ols":
        return "OLS"
    elif method == "2sls":
        return "2SLS"
    elif method.endswith("lasso") or "lasso" in method:
        return "LASSO"
    elif "2sls" in method:
        return "2SLS"
    elif "ols" in method:
        return "OLS"
    else:
        return method.upper()

combined["first_stage"] = combined["method"].apply(get_first_stage)

for fs in sorted(combined["first_stage"].unique()):
    subset = combined[combined["first_stage"] == fs]

    plt.figure(figsize=(15, 6))
    sns.barplot(data=subset, x="label", y="mse", hue="method")
    plt.title(f"MSE by Model type ({fs}) and Condition ")
    plt.xlabel("Condition (n-L-context)")
    plt.ylabel("MSE")
    plt.xticks(rotation=90)
    plt.legend(
        bbox_to_anchor=(1.05, 1),
        loc="upper left",
        borderaxespad=0.
    )
    plt.tight_layout()

    plt.savefig(f"plots/mse-{fs}.png")
    plt.close()
