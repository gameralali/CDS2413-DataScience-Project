# ============================================================
# CDS 2413 – Introduction to Data Science
# CLO3 – Part 4: Cluster Analysis (K-Means)
# Dataset: Cancer Patient Dataset
# ============================================================

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.preprocessing import LabelEncoder
from sklearn.impute import SimpleImputer

# ── Load and Prepare Data ─────────────────────────────────────
dataset = pd.read_excel('cancer_patient_dataset_readable_final.xlsx')

X = dataset[['Age', 'Obesity Level']].values

# Handle missing values
imputer = SimpleImputer(strategy='most_frequent')
X = imputer.fit_transform(X)

# Encode if needed
le = LabelEncoder()
for i in range(X.shape[1]):
    try:
        X[:, i] = X[:, i].astype(float)
    except ValueError:
        X[:, i] = le.fit_transform(X[:, i])

X = X.astype(float)

# ── Elbow Method ──────────────────────────────────────────────
wcss = []
for i in range(1, 11):
    kmeans = KMeans(n_clusters=i, init='k-means++', random_state=42, n_init=10)
    kmeans.fit(X)
    wcss.append(kmeans.inertia_)

plt.figure(figsize=(8, 5))
plt.plot(range(1, 11), wcss, marker='o', color='steelblue')
plt.title('The Elbow Method')
plt.xlabel('Number of Clusters')
plt.ylabel('WCSS')
plt.tight_layout()
plt.savefig("elbow_method.png")
plt.show()

# ── K-Means Clustering (k=3) ──────────────────────────────────
kmeans = KMeans(n_clusters=3, init='k-means++', random_state=42, n_init=10)
y_kmeans = kmeans.fit_predict(X)

print("Cluster assignments (first 20):", y_kmeans[:20])
print("\nCluster sizes:")
for i in range(3):
    print(f"  Cluster {i+1}: {np.sum(y_kmeans == i)} patients")

# ── Visualisation ─────────────────────────────────────────────
plt.figure(figsize=(8, 6))
plt.scatter(X[y_kmeans == 0, 0], X[y_kmeans == 0, 1],
            s=80, c='red',   label='Cluster 1 – Low Risk',    alpha=0.6)
plt.scatter(X[y_kmeans == 1, 0], X[y_kmeans == 1, 1],
            s=80, c='blue',  label='Cluster 2 – Medium Risk', alpha=0.6)
plt.scatter(X[y_kmeans == 2, 0], X[y_kmeans == 2, 1],
            s=80, c='green', label='Cluster 3 – High Risk',   alpha=0.6)
plt.scatter(kmeans.cluster_centers_[:, 0],
            kmeans.cluster_centers_[:, 1],
            s=200, c='black', marker='X', label='Centroids')
plt.title('Clusters of Patients (Cancer Dataset)')
plt.xlabel('Age')
plt.ylabel('Obesity Level')
plt.legend()
plt.tight_layout()
plt.savefig("kmeans_clusters.png")
plt.show()

# ── Strategy Formulation ──────────────────────────────────────
print("\n=== Strategy Formulation ===")
print("Cluster 1 – Low Risk Patients")
print("  Strategy: Maintain a healthy lifestyle with regular exercise and balanced diet.")
print()
print("Cluster 2 – Medium Risk Patients")
print("  Strategy: Improve lifestyle habits and reduce risk factors such as smoking and pollution exposure.")
print()
print("Cluster 3 – High Risk Patients")
print("  Strategy: Regular medical checkups, early screening, and targeted medical care.")
