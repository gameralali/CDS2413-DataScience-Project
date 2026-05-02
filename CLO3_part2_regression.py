# ============================================================
# CDS 2413 – Introduction to Data Science
# CLO3 – Part 2: Regression Models
# Dataset: Cancer Patient Dataset
# ============================================================

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet
from sklearn.preprocessing import PolynomialFeatures
from sklearn.metrics import r2_score

# ── Load and Prepare Data ─────────────────────────────────────
data = pd.read_excel('cancer_patient_dataset_readable_final.xlsx')

# Handle missing values
data['Age'] = data['Age'].fillna(data['Age'].mean())
data['Obesity Level'] = data['Obesity Level'].fillna(data['Obesity Level'].mode()[0])
data['Cancer Risk Level'] = data['Cancer Risk Level'].fillna(data['Cancer Risk Level'].mode()[0])

# Encode target variable
le = LabelEncoder()
data['Cancer Risk Level'] = le.fit_transform(data['Cancer Risk Level'])

# Remove outliers using IQR on Age
Q1 = data['Age'].quantile(0.25)
Q3 = data['Age'].quantile(0.75)
IQR = Q3 - Q1
lower = Q1 - 1.5 * IQR
upper = Q3 + 1.5 * IQR
data = data[(data['Age'] >= lower) & (data['Age'] <= upper)]

# ── Define Features and Target ────────────────────────────────
X = data[['Age', 'Obesity Level']]
y = data['Cancer Risk Level']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=0
)

# ── 1. Linear Regression ──────────────────────────────────────
regressor1 = LinearRegression()
regressor1.fit(X_train, y_train)
y_pred1 = regressor1.predict(X_test)
r2_linear = r2_score(y_test, y_pred1)
print(f"Linear Regression     R² = {r2_linear:.4f}")

# ── 2. Polynomial Regression (Degree 2) ───────────────────────
poly_reg = PolynomialFeatures(degree=2)
X_poly   = poly_reg.fit_transform(X_train)
regressor2 = LinearRegression()
regressor2.fit(X_poly, y_train)
y_pred2 = regressor2.predict(poly_reg.transform(X_test))
r2_poly = r2_score(y_test, y_pred2)
print(f"Polynomial Regression R² = {r2_poly:.4f}")

# ── 3. Ridge Regression ───────────────────────────────────────
regressor3 = Ridge(alpha=1.0)
regressor3.fit(X_train, y_train)
y_pred3 = regressor3.predict(X_test)
r2_ridge = r2_score(y_test, y_pred3)
print(f"Ridge Regression      R² = {r2_ridge:.4f}")

# ── 4. Lasso Regression ───────────────────────────────────────
regressor4 = Lasso(alpha=0.1)
regressor4.fit(X_train, y_train)
y_pred4 = regressor4.predict(X_test)
r2_lasso = r2_score(y_test, y_pred4)
print(f"Lasso Regression      R² = {r2_lasso:.4f}")

# ── 5. Elastic Net Regression ─────────────────────────────────
regressor5 = ElasticNet(alpha=0.1)
regressor5.fit(X_train, y_train)
y_pred5 = regressor5.predict(X_test)
r2_elastic = r2_score(y_test, y_pred5)
print(f"Elastic Net           R² = {r2_elastic:.4f}")

# ── Best Model Summary ────────────────────────────────────────
scores = {
    'Linear':      r2_linear,
    'Polynomial':  r2_poly,
    'Ridge':       r2_ridge,
    'Lasso':       r2_lasso,
    'Elastic Net': r2_elastic,
}
best_model = max(scores, key=scores.get)
print(f"\nBest Model: {best_model} (R² = {scores[best_model]:.4f})")

# ── Regression Equation (Linear) ─────────────────────────────
intercept = regressor1.intercept_
coef      = regressor1.coef_
print(f"\nLinear Regression Equation:")
print(f"  Cancer Risk Level = {intercept:.4f} + {coef[0]:.4f}(Age) + {coef[1]:.4f}(Obesity Level)")

# ── Visualisation: Actual vs Predicted ───────────────────────
plt.figure(figsize=(8, 5))
plt.scatter(y_test, y_pred1, color='steelblue', alpha=0.6, label='Predicted vs Actual')
plt.plot([y_test.min(), y_test.max()],
         [y_test.min(), y_test.max()], 'r--', label='Perfect Fit')
plt.xlabel("Actual Cancer Risk Level")
plt.ylabel("Predicted Cancer Risk Level")
plt.title("Actual vs Predicted – Linear Regression")
plt.legend()
plt.tight_layout()
plt.savefig("regression_actual_vs_predicted.png")
plt.show()
