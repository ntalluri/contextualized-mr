import numpy as np
import pandas as pd

# TODO add code to make the data folder
# and set a folder var to use for input data and output data


def simulate_one_sample_MR(
    n=5000, # the number of individuals

    L=15, # the number of SNPs (instruments)

    beta=0.30, # the true causal effect of X on Y

    pi_scale=0.10,  # instrument strength scale for the magnitude of SNP -> exposure 
    # Each SNP G has an effect pi on the exposure X. The larger pi_scale, the stronger the SNPs explain X

    # Exclusion restriction: Instruments affect Y only through X.
    # removed pleio_mu and pleio_tau, so SNPs don’t have direct effects on Y
    # pleio_mu=0.0, pleio_tau=0.0, 
    # # TODO figure out if I want these values?
    # # set tau>0 to add horizontal pleiotropy: theta ~ N(mu, tau^2) 
    # # defines horizontal pleiotropy that is the direct SNP -> Y effects that break exclusion restriction
    # # for MR to be under under the ideal IV assumptions, these are set to 0 s.t  all exclusion restriction assumptions hold (the only pathway from SNPs to outcome is through the exposure)
    # # MR can sometimes handle violations like pleiotropy. To see how robust the method is, I deliberately turn these knobs

    sigma_x=1.0, sigma_y=1.0, # noise in exposure and outcome

    seed=7, # reproducibility

    use_context=False, # toggle if data will include context in generate data
    
    # Coefs for context in X and Y for how important they are
    # gamma_c: effects of context on exposure X
    gamma_c = np.array([
        0.03,  # age
        0.05,  # sex
        0.14,  # bmi
        0.16,  # smoker
        0.00,  # height
        0.00,  # weight
        0.08,  # bp_sys
        0.02,  # cholesterol
        0.12,  # glucose
        -0.06  # sleep (more sleep -> lower exposure)
    ]),
    # zeta_c : effects of context on outcome Y
    zeta_c = np.array([
        0.02,  # age
        0.03,  # sex
        0.10,  # bmi
        0.05,  # smoker
        0.00,  # height
        0.00,  # weight
        0.12,  # bp_sys
        0.15,  # cholesterol
        0.18,  # glucose
        0.04   # sleep
    ])

):
    """
    Simulate individual level GWAS style data for a one-sample Mendelian Randomization

    If use_context=False: baseline MR data
        X = Gπ + U + εx
        Y = βX + Gθ + U + εy
    
    If use_context=True: realistic contextualized DGP
        X = Gπ + Cγ + U + εx
        Y = βX + Gθ + Cζ + U + εy

    Returns:
    df  : DataFrame with columns [Y, X, U, age, sex, G0..G{L-1}]
        Columns:
        - X : simulated exposure phenotype (e.g., BMI, cholesterol).
        - Y : simulated outcome phenotype (e.g., blood pressure, disease risk).
        - 'age','sex': measured covariates (optional; useful for context MR).
        - G0..G{L-1}: SNP genotypes for each individual.
            - Values are 0/1/2 = number of minor alleles at that SNP.

    meta : dict with truth and meta data (beta, U, pi, mafs, L, n, C (standardized context))
    """
    rng = np.random.default_rng(seed) 
    # keeps all randomness tied to the one generator
    # every random draw in the code (mafs, genotypes, covariates, confounders, error terms) uses this rng
    # this means the entire simulated cohort is reproducible by just the single seed
   
    # genotypes: independent SNPs with minor allele frequency in [0.05, 0.5]
    # a conventional metric used by geneticists to quantify the variability of a SNP within a population.
    # It provides a standardized way to describe how common or rare a particular SNP variant is
    # 0.5 = both alleles equally frequent (the "minor" allele cannot be more common than 50%), near 0 = rare variant.
    mafs = rng.uniform(0.05, 0.5, size=L)

    # drawing from Binomial(2, MAF) mimics how alleles are inherited 
    # two draws per person, each with probability = MAF
    G = rng.binomial(2, mafs, size=(n, L))  # n x L
    # 0 = no copies of the minor allele (homozygous major)
    # 1 = one copy (heterozygous)
    # 2 = two copies (homozygous minor)
    # all the SNPs simulated have an effect

    # Unmeasured confounder that affects both X and Y
    # By adding U, I create a situation where X and Y are spuriously correlated (confounding)
    U = rng.normal(0, 1, size=n)

    # Independence (exchangeability): Instruments are independent of unmeasured confounders. by construction, G is independent from U and Y

    # Context
    # want continous value context will do better
    age = rng.integers(12, 50, size=n)
    sex = rng.integers(0, 2, size=n) # 0/1 0 female, 1 male
    bmi = rng.normal(25, 4, size=n)  
    smoker = rng.integers(0, 2, size=n) # 0/1 0 no, 1 yes
    height = rng.normal(170, 10, size=n)   # cm
    weight = rng.normal(70, 15, size=n) # kg
    bp_sys = rng.normal(120, 15, size=n) # systolic blood pressure
    cholesterol = rng.normal(200, 40, size=n)  # mg/dL
    glucose = rng.normal(90, 15, size=n)   # fasting glucose mg/dL
    sleep = rng.normal(7, 1.5, size=n)

    # stack in a fixed order
    C_raw = np.column_stack([
        age, sex, bmi, smoker, height, weight, bp_sys, cholesterol, glucose, sleep
    ])
    # standardize columns so coefficients are on a comparable scale
    C_mean = C_raw.mean(axis=0)
    C_std  = C_raw.std(axis=0, ddof=0)
    C = (C_raw - C_mean) / C_std

    # the context stacked if not doing standarization
    # C = np.column_stack([age, sex, bmi, smoker, height, weight, bp_sys, cholesterol, glucose, sleep])

    # Instrument effects on X
    pi = rng.normal(0, pi_scale, size=L) 
    # using pi > 0 will lead to relevance: instruments (G) are associated with the exposure (X).
    # but this is picking at random how weak or strong that association is

    # Horizontal pleiotropy effects on Y (theta); 0 means exclusion restriction holds
    # if pleio_mu != 0.0 or pleio_tau != 0.0:
    #     theta = rng.normal(pleio_mu, pleio_tau, size=L)
    # else:
    #     theta = np.zeros(L)

    # exposure X
    v = rng.normal(0, sigma_x, size=n) # noise
    # point-estimate-identifying condition is held by a constant beta
    if use_context:
        X = (G @ pi) + (C @ gamma_c) + U + v # TODO: figure out context formalation
    else:
        X = (G @ pi) + (U + v) # TODO: figure out if I need to have the same formulation as the context formulation

    # outcome Y
    u = rng.normal(0, sigma_y, size=n) # noise
    if use_context:
        Y = beta * X + (C @ zeta_c) + U + u  # TODO: figure out context formalation
    else:
        Y = (beta * X) + (U + u) # TODO: figure out if I need to have the same formulation as the context formulation
         # want continous value outcome will do better (converges faster)

    df = pd.DataFrame({
        "Y": Y, "X": X,
        "age": age, "sex": sex, "bmi": bmi, "smoker": smoker, "height": height, "weight": weight, "bp_sys": bp_sys, "cholesterol": cholesterol, "glucose": glucose, "sleep": sleep
    })
    for j in range(L):
        df[f"G{j}"] = G[:, j]

    meta_data = dict(beta_true=beta, U = U, pi=pi, mafs=mafs, L=L, n=n, C=C)
    
    return df, meta_data

# individuals = 5000
# l = 15
# context = True

s = 7
b = 0.3
pi = 0.1
err_x = 1.0
err_y = 1.0
gamma_c = np.array([
        0.03,  # age
        0.05,  # sex
        0.14,  # bmi
        0.16,  # smoker
        0.00,  # height
        0.00,  # weight
        0.08,  # bp_sys
        0.02,  # cholesterol
        0.12,  # glucose
        -0.06  # sleep (more sleep -> lower exposure)
    ])
zeta_c = np.array([
        0.02,  # age
        0.03,  # sex
        0.10,  # bmi
        0.05,  # smoker
        0.00,  # height
        0.00,  # weight
        0.12,  # bp_sys
        0.15,  # cholesterol
        0.18,  # glucose
        0.04   # sleep
    ])


individuals_list = [1000, 3000, 5000, 10000, 20000]
L_list = [5, 10, 15, 30, 50]
contexts = [True, False]

for individuals in individuals_list:
    for l in L_list:
        for context in contexts:

            df, meta = simulate_one_sample_MR(
                    n=individuals, L=l, beta=b, pi_scale=pi,
                    sigma_x=err_x, sigma_y=err_y, seed=s, 
                    use_context=context, gamma_c=gamma_c, zeta_c= zeta_c)


            df.to_csv(f"data/GWAS-MR-{individuals}-{l}-{b}-{context}-{s}.csv", sep="\t", index=False)

            with open(f"data/metadata-{individuals}-{l}-{b}-{context}-{s}.txt", "w") as f:
                for k, v in  meta.items():
                    f.write(f"{k}: {v}\n")
