# ============================================================
# CDS 2413 – Introduction to Data Science
# CLO3 – Part 3: Classification Models
# Dataset: Cancer Patient Dataset
# ============================================================

import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier

# ── Load and Preprocess Data ──────────────────────────────────
dataset = pd.read_excel('cancer_patient_dataset_readable_final.xlsx')

X = dataset[['Age', 'Obesity Level', 'Smoking Level', 'Air Pollution Exposure']].values
y = dataset['Cancer Risk Level'].values

# Handle missing values
imputer = SimpleImputer(strategy='most_frequent')
X = imputer.fit_transform(X)

# Encode features
le_X = LabelEncoder()
for i in range(X.shape[1]):
    X[:, i] = le_X.fit_transform(X[:, i])

# Encode target variable (Low=0, Medium=1, High=2)
le_y = LabelEncoder()
y = le_y.fit_transform(y)

# Split and scale
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=1
)
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test  = sc.transform(X_test)

# ── Target Variable Distribution ──────────────────────────────
print("=== Target Variable Distribution ===")
print(f"  Low Risk    (0): {np.sum(y == 0)}")
print(f"  Medium Risk (1): {np.sum(y == 1)}")
print(f"  High Risk   (2): {np.sum(y == 2)}")

# ── 1. Logistic Regression ────────────────────────────────────
model1 = LogisticRegression(max_iter=1000)
model1.fit(X_train, y_train)
y_pred1 = model1.predict(X_test)
print("\n=== Logistic Regression ===")
print(classification_report(y_test, y_pred1))
print("Accuracy:", round(accuracy_score(y_test, y_pred1), 4))

# ── 2. K-Nearest Neighbours ───────────────────────────────────
model2 = KNeighborsClassifier(n_neighbors=5)
model2.fit(X_train, y_train)
y_pred2 = model2.predict(X_test)
print("\n=== KNN ===")
print(classification_report(y_test, y_pred2))
print("Accuracy:", round(accuracy_score(y_test, y_pred2), 4))

# ── 3. Support Vector Machine ─────────────────────────────────
model3 = SVC()
model3.fit(X_train, y_train)
y_pred3 = model3.predict(X_test)
print("\n=== SVM ===")
print(classification_report(y_test, y_pred3))
print("Accuracy:", round(accuracy_score(y_test, y_pred3), 4))

# ── 4. Decision Tree ──────────────────────────────────────────
model4 = DecisionTreeClassifier(random_state=1)
model4.fit(X_train, y_train)
y_pred4 = model4.predict(X_test)
print("\n=== Decision Tree ===")
print(classification_report(y_test, y_pred4))
print("Accuracy:", round(accuracy_score(y_test, y_pred4), 4))

# ── 5. Random Forest ──────────────────────────────────────────
model5 = RandomForestClassifier(random_state=1)
model5.fit(X_train, y_train)
y_pred5 = model5.predict(X_test)
print("\n=== Random Forest ===")
print(classification_report(y_test, y_pred5))
print("Accuracy:", round(accuracy_score(y_test, y_pred5), 4))

# ── Accuracy Summary ──────────────────────────────────────────
print("\n=== Accuracy Summary ===")
models = {
    'Logistic Regression': accuracy_score(y_test, y_pred1),
    'KNN':                 accuracy_score(y_test, y_pred2),
    'SVM':                 accuracy_score(y_test, y_pred3),
    'Decision Tree':       accuracy_score(y_test, y_pred4),
    'Random Forest':       accuracy_score(y_test, y_pred5),
}
for name, acc in sorted(models.items(), key=lambda x: x[1], reverse=True):
    print(f"  {name:<22}: {acc:.4f}")

best = max(models, key=models.get)
print(f"\nBest Model: {best} ({models[best]:.4f})")
