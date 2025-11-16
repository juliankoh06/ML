"""
Exploratory Data Analysis (EDA)  for adult.csv

"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os

# Create a directory to save the figures
output_dir = 'figures/eda'
os.makedirs(output_dir, exist_ok=True)

# Seaborn style
sns.set_theme(style="whitegrid")


# Function to Save Plots
def save_plot(fig, filename):
    filepath = os.path.join(output_dir, filename)
    fig.savefig(filepath, bbox_inches='tight')
    print(f"Saved plot: {filepath}")
    plt.close(fig)

# 1. ADULT INCOME DATASET (CLASSIFICATION)

print("\n" + "="*80)
print(" EXPLORATORY DATA ANALYSIS: ADULT INCOME DATASET ")
print("="*80)

try:
    df_adult = pd.read_csv('data/adult.csv')

    # --- 1.1 Initial Data Analysis ---
    print("\n--- 1.1 Initial Data Analysis ---")
    
    # --- NEW: Replace '?' with NaN to find real missing values ---
    print("\n[INFO] Replacing '?' with NaN...")
    df_adult.replace('?', np.nan, inplace=True)
    
    print("\n[INFO] First 5 rows:")
    print(df_adult.head().to_string())
    
    print("\n[INFO] Data types and memory usage:")
    df_adult.info()
    
    print("\n[INFO] Checking for REAL missing values (after replacing '?'):")
    print(df_adult.isnull().sum())
    
    print("\n[INFO] Checking for duplicated rows:")
    print(f"Found {df_adult.duplicated().sum()} duplicated rows.")
    
    print("\n[INFO] Summary statistics for numerical features:")
    print(df_adult.describe().to_string())
    
    print("\n[INFO] Summary statistics for categorical features:")
    print(df_adult.describe(include='object').to_string())

    # --- 1.2 Visualizations ---
    print("\n--- 1.2 Generating Visualizations ---")

    # Target variable distribution ('income')
    fig, ax = plt.subplots(figsize=(8, 6))
    sns.countplot(data=df_adult, x='income', ax=ax, palette='viridis')
    ax.set_title('Distribution of Income Classes')
    ax.set_xlabel('Income')
    ax.set_ylabel('Count')
    save_plot(fig, 'adult_income_distribution.png')

    # Age distribution
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.histplot(data=df_adult, x='age', hue='income', kde=True, multiple='stack', palette='coolwarm')
    ax.set_title('Age Distribution by Income')
    ax.set_xlabel('Age')
    ax.set_ylabel('Count')
    save_plot(fig, 'adult_age_distribution.png')

    # Box plot of Age vs Income
    fig, ax = plt.subplots(figsize=(8, 6))
    sns.boxplot(data=df_adult, x='income', y='age', ax=ax, palette='viridis')
    ax.set_title('Age vs. Income')
    ax.set_xlabel('Income')
    ax.set_ylabel('Age')
    save_plot(fig, 'adult_age_vs_income.png')

    # Education vs Income
    fig, ax = plt.subplots(figsize=(12, 8))
    sns.countplot(data=df_adult, y='education', hue='income', palette='magma', order=df_adult['education'].value_counts().index)
    ax.set_title('Education Level vs. Income')
    ax.set_xlabel('Count')
    ax.set_ylabel('Education Level')
    plt.tight_layout()
    save_plot(fig, 'adult_education_vs_income.png')

    # Workclass vs Income
    fig, ax = plt.subplots(figsize=(12, 8))
    sns.countplot(data=df_adult, y='workclass', hue='income', palette='plasma', order=df_adult['workclass'].value_counts().index)
    ax.set_title('Work Class vs. Income')
    ax.set_xlabel('Count')
    ax.set_ylabel('Work Class')
    plt.tight_layout()
    save_plot(fig, 'adult_workclass_vs_income.png')

    # Sex vs Income (for bias analysis) ---
    fig, ax = plt.subplots(figsize=(8, 6))
    sns.countplot(data=df_adult, x='sex', hue='income', palette='coolwarm')
    ax.set_title('Sex vs. Income')
    ax.set_xlabel('Sex')
    ax.set_ylabel('Count')
    save_plot(fig, 'adult_sex_vs_income.png')

    # Race vs Income (for bias analysis) ---
    fig, ax = plt.subplots(figsize=(12, 8))
    sns.countplot(data=df_adult, y='race', hue='income', palette='plasma', order=df_adult['race'].value_counts().index)
    ax.set_title('Race vs. Income')
    ax.set_xlabel('Count')
    ax.set_ylabel('Race')
    plt.tight_layout()
    save_plot(fig, 'adult_race_vs_income.png')

    # Correlation heatmap for numerical features
    numeric_cols_adult = df_adult.select_dtypes(include=np.number).columns
    fig, ax = plt.subplots(figsize=(12, 8))
    sns.heatmap(df_adult[numeric_cols_adult].corr(), annot=True, fmt='.2f', cmap='coolwarm', ax=ax)
    ax.set_title('Correlation Heatmap of Numerical Features (Adult Dataset)')
    save_plot(fig, 'adult_correlation_heatmap.png')

except FileNotFoundError:
    print("\n[ERROR] 'data/adult.csv' not found. Skipping analysis for this dataset.")