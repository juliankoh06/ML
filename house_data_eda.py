"""
Exploratory Data Analysis (EDA) Script 

"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os


# Configuration
output_dir = 'figures/eda'
os.makedirs(output_dir, exist_ok=True)

# Seaborn style
sns.set_theme(style="whitegrid")

# Save Plots

def save_plot(fig, filename):
    """Saves a matplotlib figure to the output directory."""
    filepath = os.path.join(output_dir, filename)
    fig.savefig(filepath, bbox_inches='tight')
    print(f" Saved plot: {filepath}")
    plt.close(fig)

#  HOUSING PRICES DATASET (REGRESSION)

print("\n" + "="*80)
print(" EXPLORATORY DATA ANALYSIS: HOUSING PRICES DATASET ('housing.csv')")
print("="*80)

try:
    df_housing = pd.read_csv('data/housing.csv')

    # --- 2.1 Initial Data Analysis ---
    print("\n--- 2.1 Initial Data Analysis ---")
    print("\n[INFO] First 5 rows:")
    print(df_housing.head().to_string())
    
    print("\n[INFO] Data types and memory usage:")
    df_housing.info()
    
    print("\n[INFO] Checking for missing values:")
    print(df_housing.isnull().sum())
    
    print("\n[INFO] Checking for duplicated rows:")
    print(f"Found {df_housing.duplicated().sum()} duplicated rows.")
    
    print("\n[INFO] Summary statistics for numerical features:")
    print(df_housing.describe().to_string())
    
    print("\n[INFO] Summary statistics for categorical features:")
    print(df_housing.describe(include='object').to_string())

    # --- 2.2 Visualizations ---
    print("\n--- 2.2 Generating Visualizations ---")

    # Target variable distribution ('median_house_value')
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.histplot(data=df_housing, x='median_house_value', kde=True, color='skyblue')
    ax.set_title('Distribution of Median House Value (Target)')
    ax.set_xlabel('Median House Value')
    ax.set_ylabel('Count')
    ax.axvline(500001, color='red', linestyle='--', label='Capped Value (500001)')
    ax.legend()
    save_plot(fig, 'housing_price_distribution.png')

    # Geographical distribution of house prices
    fig, ax = plt.subplots(figsize=(12, 10))
    scatter = ax.scatter(df_housing['longitude'], df_housing['latitude'], alpha=0.4,
                         s=df_housing['population']/100, label='population',
                         c=df_housing['median_house_value'], cmap=plt.get_cmap('jet'))
    cbar = fig.colorbar(scatter)
    cbar.set_label('Median House Value')
    ax.set_title('Geographical Distribution of House Prices')
    ax.set_xlabel('Longitude')
    ax.set_ylabel('Latitude')
    ax.legend()
    save_plot(fig, 'housing_geo_distribution.png')

    # --- NEW: Median Income vs. Price (Strongest Predictor) ---
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.scatterplot(data=df_housing, x='median_income', y='median_house_value', 
                    alpha=0.2, color='darkgreen')
    ax.set_title('Median Income vs. Median House Value')
    ax.set_xlabel('Median Income (in 10,000s)')
    ax.set_ylabel('Median House Value')
    save_plot(fig, 'housing_income_vs_price.png')

    # Ocean Proximity vs Median House Value
    fig, ax = plt.subplots(figsize=(12, 8))
    sns.boxplot(data=df_housing, x='ocean_proximity', y='median_house_value', palette='viridis')
    ax.set_title('Median House Value by Ocean Proximity')
    ax.set_xlabel('Ocean Proximity')
    ax.set_ylabel('Median House Value')
    save_plot(fig, 'housing_ocean_proximity_vs_price.png')

    # --- NEW: Histograms for Skewed Features (Justifies Ratios) ---
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    fig.suptitle('Distributions of Skewed Features (Justifying Ratios)')
    
    sns.histplot(df_housing['total_rooms'], kde=False, ax=axes[0], bins=50)
    axes[0].set_title('Total Rooms per Block (Skewed)')
    
    sns.histplot(df_housing['population'], kde=False, ax=axes[1], bins=50)
    axes[1].set_title('Population per Block (Skewed)')
    
    sns.histplot(df_housing['households'], kde=False, ax=axes[2], bins=50)
    axes[2].set_title('Households per Block (Skewed)')
    
    plt.tight_layout()
    save_plot(fig, 'housing_skewed_features.png')

    # Correlation heatmap for numerical features
    numeric_cols_housing = df_housing.select_dtypes(include=np.number).columns
    fig, ax = plt.subplots(figsize=(14, 10))
    sns.heatmap(df_housing[numeric_cols_housing].corr(), annot=True, fmt='.2f', cmap='coolwarm', ax=ax)
    ax.set_title('Correlation Heatmap of Numerical Features (Housing Dataset)')
    save_plot(fig, 'housing_correlation_heatmap.png')

except FileNotFoundError:
    print("\n[ERROR] 'data/housing.csv' not found. Skipping analysis for this dataset.")

print("\n" + "="*80)
print("EDA Script Finished")
print(f"All plots have been saved to the '{output_dir}' directory.")
print("="*80)