#import "@preview/academic-alt:0.1.0": *


#show: university-assignment.with(
  title: "Data Mining Final Practical File",
  subtitle: "24/48029",
  author: "Vivaan Singh Adhikari",
  details: (
    course: "Data Mining DSE",
    instructor: "Prof. Archana Gahalaut",
    hardware: "No specifications",
    software: "Python, Pandas, Typst(documentation)",
  )
)

= Introduction

This assignment entails my solutions to the question assigned as per the course's guidelines. 
All the final files are available on https://github.com/user7537/coursework/


== Code

Q1: Apply data cleaning techniques on any dataset (e.g., Paper Reviews dataset in UCI repository). Techniques may include handling missing values, outliers and inconsistent values. A set of validation rules can be prepared based on the dataset and validations can be performed.

```python
import pandas as pd
import numpy as np

colnames = ["A1","A2","A3","A4","A5","A6","A7","A8","A9","A10",
            "A11","A12","A13","A14","A15","class"]

df = pd.read_csv("japanese_credit_screening/crx.data",
                 names=colnames,
                 na_values="?")

# Numeric columns
num_cols = df.select_dtypes(include=["float64","int64"]).columns
cat_cols = df.select_dtypes(include=["object"]).columns

# Fix numeric missing with median
df[num_cols] = df[num_cols].apply(lambda col: col.fillna(col.median()))

# Fix categorical missing with mode
for c in cat_cols:
    df[c] = df[c].fillna(df[c].mode()[0])

# Outlier removal (IQR)
for col in num_cols:
    Q1 = df[col].quantile(0.25)
    Q3 = df[col].quantile(0.75)
    IQR = Q3 - Q1
    df = df[(df[col] >= Q1 - 1.5*IQR) & (df[col] <= Q3 + 1.5*IQR)]

df.to_csv("q1_cleaned.csv", index=False)
print("Saved q1_cleaned.csv")

```


