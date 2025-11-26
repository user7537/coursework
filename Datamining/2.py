import pandas as pd
from sklearn.preprocessing import StandardScaler, MinMaxScaler, KBinsDiscretizer

colnames = ["A1","A2","A3","A4","A5","A6","A7","A8","A9","A10",
            "A11","A12","A13","A14","A15","class"]

df = pd.read_csv("japanese_credit_screening/crx.data",
                 names=colnames,
                 na_values="?")

num_cols = df.select_dtypes(include=["float64","int64"]).columns

# Standardization
std_df = df.copy()
std_df[num_cols] = StandardScaler().fit_transform(std_df[num_cols])
std_df.to_csv("q2_standardized.csv", index=False)

# Normalization
norm_df = df.copy()
norm_df[num_cols] = MinMaxScaler().fit_transform(norm_df[num_cols])
norm_df.to_csv("q2_normalized.csv", index=False)

# Discretization
disc_df = df.copy()
disc = KBinsDiscretizer(n_bins=3, encode="ordinal", strategy="uniform")
disc_df[num_cols] = disc.fit_transform(disc_df[num_cols])
disc_df.to_csv("q2_discretized.csv", index=False)

# Sampling 30%
sample_df = df.sample(frac=0.3, random_state=42)
sample_df.to_csv("q2_sampled.csv", index=False)

print("Generated all q2 outputs")


