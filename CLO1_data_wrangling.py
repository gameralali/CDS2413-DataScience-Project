# ============================================================
# CDS 2413 – Introduction to Data Science
# CLO1: Introduction and Data Wrangling
# Dataset: Cancer Patient Dataset
# ============================================================

import pandas as pd
import numpy as np

# ── Load Dataset ─────────────────────────────────────────────
df = pd.read_excel('cancer_patient_dataset_readable_final.xlsx')

print("Shape:", df.shape)
print("\nFirst 5 rows:")
print(df.head())

print("\nColumn names:")
print(df.columns.tolist())

print("\nData types:")
print(df.dtypes)

# ── Variables ────────────────────────────────────────────────
# Dependent Variable:
#   Cancer Risk Level
#
# Independent Variables:
#   Age, Gender, Smoking Level, Air Pollution Level,
#   Alcohol Consumption Level, Symptoms

# ── Sampling ─────────────────────────────────────────────────

# Random Sampling (n=150)
random_sample = df.sample(n=150, random_state=1)
print("\nRandom Sample Shape:", random_sample.shape)

# Systematic Sampling (every 5th row)
systematic_sample = df.iloc[::5]
print("Systematic Sample Shape:", systematic_sample.shape)
