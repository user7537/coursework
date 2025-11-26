import pandas as pd
 
df = pd.read_csv("japanese_credit_screening/crx.data")
# Missing values
df = df.fillna(df.median(numeric_only=True))
df = df.fillna(df.mode().iloc[0])

# Outlier removal using IQR
for col in df.select_dtypes(include="number").columns:
    Q1 = df[col].quantile(0.25)
    Q3 = df[col].quantile(0.75)
    IQR = Q3 - Q1
    df = df[(df[col] >= Q1 - 1.5*IQR) & (df[col] <= Q3 + 1.5*IQR)]

# Inconsistent values example (domain-based)
# Example: remove negative ages
if "age" in df.columns:
    df = df[df["age"] >= 0]

df.to_csv("q1_cleaned.csv", index=False)
print("Done. Saved as q1_cleaned.csv")

