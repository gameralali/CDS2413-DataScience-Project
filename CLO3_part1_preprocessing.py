# ============================================================
# CDS 2413 – Introduction to Data Science
# CLO3 – Part 1: Data Preprocessing
# Dataset: Cancer Patient Dataset
# ============================================================

import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split

# ── 1. Data Collection ────────────────────────────────────────
dataset = pd.read_excel('cancer_patient_dataset_readable_final.xlsx')

X = dataset[['Age', 'Obesity Level', 'Smoking Level', 'Air Pollution Exposure']].values
y = dataset['Cancer Risk Level'].values

print("X (first 5 rows):")
print(X[:5])
print("\ny (first 5 values):")
print(y[:5])

# ── 2. Data Cleaning – Handle Missing Values ──────────────────
imputer = SimpleImputer(strategy='most_frequent')
X = imputer.fit_transform(X)

# ── 3. Encode Categorical Variables ──────────────────────────
le_X = LabelEncoder()
for i in range(X.shape[1]):
    X[:, i] = le_X.fit_transform(X[:, i])

le_y = LabelEncoder()
y = le_y.fit_transform(y)

print("\nEncoded X (first 5 rows):")
print(X[:5])
print("\nEncoded y (first 5 values):")
print(y[:5])

# ── 4. Split Dataset (80% train / 20% test) ───────────────────
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=1
)

print(f"\nTraining set size: {X_train.shape[0]}")
print(f"Test set size:     {X_test.shape[0]}")

# ── 5. Feature Scaling (Standardisation) ─────────────────────
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test  = sc.transform(X_test)

print("\nX_train after scaling (first 5 rows):")
print(X_train[:5])
