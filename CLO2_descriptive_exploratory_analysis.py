# ============================================================
# CDS 2413 – Introduction to Data Science
# CLO2: Descriptive, Exploratory and Statistical Analysis
# Dataset: Cancer Patient Dataset
# ============================================================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import pearsonr, spearmanr, chi2_contingency
from scipy import stats

# ── Load Dataset ─────────────────────────────────────────────
df = pd.read_excel('cancer_patient_dataset_readable_final.xlsx')

# ── Descriptive Statistics Function ──────────────────────────
def descriptive_stats(dataframe, column):
    return dataframe[column].describe()

# ── Random Sampling ──────────────────────────────────────────
random_sample = df.sample(n=150, random_state=1)
print("=== Descriptive Stats – Random Sample ===")
print(descriptive_stats(random_sample, 'Cancer Risk Level'))

# ── Systematic Sampling ───────────────────────────────────────
systematic_sample = df.iloc[::5]
print("\n=== Descriptive Stats – Systematic Sample ===")
print(descriptive_stats(systematic_sample, 'Cancer Risk Level'))

# ── Visualisations ────────────────────────────────────────────

# 1. Scatter Plot – Age vs Cancer Risk Level
plt.figure()
plt.scatter(df['Age'], df['Cancer Risk Level'], alpha=0.5)
plt.xlabel("Age")
plt.ylabel("Cancer Risk Level")
plt.title("Age vs Cancer Risk Level")
plt.tight_layout()
plt.savefig("scatter_age_vs_risk.png")
plt.show()

# 2. Box Plot – Gender vs Cancer Risk Level
plt.figure()
sns.boxplot(x=df['Gender'], y=df['Cancer Risk Level'])
plt.title("Gender vs Cancer Risk Level")
plt.tight_layout()
plt.savefig("boxplot_gender_vs_risk.png")
plt.show()

# 3. Histogram – Cancer Risk Level
plt.figure()
plt.hist(df['Cancer Risk Level'], bins=10, color='steelblue', edgecolor='black')
plt.title("Histogram of Cancer Risk Level")
plt.xlabel("Cancer Risk Level")
plt.ylabel("Frequency")
plt.tight_layout()
plt.savefig("histogram_risk_level.png")
plt.show()

# 4. Heatmap – Correlation
plt.figure()
sns.heatmap(df[['Age', 'Cancer Risk Level']].corr(numeric_only=True), annot=True, cmap='coolwarm')
plt.title("Correlation Heatmap")
plt.tight_layout()
plt.savefig("heatmap_correlation.png")
plt.show()

# ── Hypothesis Tests ──────────────────────────────────────────

# Pearson Correlation – Age vs Cancer Risk Level
pearson_result = pearsonr(df['Age'], df['Cancer Risk Level'])
print("\n=== Pearson Correlation (Age vs Cancer Risk Level) ===")
print(f"  Correlation: {pearson_result[0]:.4f}, p-value: {pearson_result[1]:.4f}")

# Spearman Correlation – Age vs Cancer Risk Level
spearman_result = spearmanr(df['Age'], df['Cancer Risk Level'])
print("\n=== Spearman Correlation (Age vs Cancer Risk Level) ===")
print(f"  Correlation: {spearman_result[0]:.4f}, p-value: {spearman_result[1]:.4f}")

# Chi-Square Test – Gender vs Cancer Risk Level
table = pd.crosstab(df['Gender'], df['Cancer Risk Level'])
chi2_result = chi2_contingency(table)
print("\n=== Chi-Square Test (Gender vs Cancer Risk Level) ===")
print(f"  Chi2: {chi2_result[0]:.4f}, p-value: {chi2_result[1]:.4f}")

# ── One Sample T-Test ─────────────────────────────────────────
ttest_result = stats.ttest_1samp(df['Cancer Risk Level'], popmean=2)
print("\n=== One Sample T-Test (Cancer Risk Level, pop mean=2) ===")
print(f"  t-statistic: {ttest_result[0]:.4f}, p-value: {ttest_result[1]:.4f}")
