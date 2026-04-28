import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

# Create a folder to save the plots (optional)
os.makedirs('plots', exist_ok=True)

# ----- Load and prepare data (same binning as preprocess) -----
filepath = 'data/dataset.csv'
df = pd.read_csv(filepath)

# Create target column
bins = [0, 3, 6, 10]
labels = ['Low', 'Medium', 'High']
df['Addiction_Level'] = pd.cut(df['Addicted_Score'], bins=bins, labels=labels, include_lowest=True, right=True)

# Keep Addicted_Score temporarily for correlation analysis (we won't use it as a feature later)
# We'll use numeric features + Addicted_Score for heatmap, but NOT for modelling.

# ----- 1. Target distribution -----
plt.figure(figsize=(6,4))
sns.countplot(data=df, x='Addiction_Level', order=['Low','Medium','High'], hue='Addiction_Level', palette='viridis', legend=False)
plt.title('Distribution of Addiction Levels')
plt.ylabel('Count')
plt.tight_layout()
plt.savefig('plots/01_target_distribution.png')
plt.show()

# ----- 2. Numeric features vs Addiction Level (boxplots) -----
numeric_features = ['Age', 'Avg_Daily_Usage_Hours', 'Sleep_Hours_Per_Night', 'Mental_Health_Score']
for col in numeric_features:
    plt.figure(figsize=(8,4))
    sns.boxplot(data=df, x='Addiction_Level', y=col, order=['Low','Medium','High'])
    plt.title(f'{col} by Addiction Level')
    plt.tight_layout()
    plt.savefig(f'plots/02_boxplot_{col}.png')
    plt.show()

# ----- 3. Categorical features vs Addiction Level (countplots) -----
categorical_features = ['Gender', 'Academic_Level', 'Most_Used_Platform',
                        'Relationship_Status', 'Affects_Academic_Performance',
                        'Conflicts_Over_Social_Media']
for col in categorical_features:
    plt.figure(figsize=(10,5))
    # Use hue to split by target
    sns.countplot(data=df, x=col, hue='Addiction_Level', order=df[col].value_counts().index)
    plt.title(f'{col} vs Addiction Level')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(f'plots/03_countplot_{col}.png')
    plt.show()

# ----- 4. Correlation heatmap (numeric + Addicted_Score) -----
# Select only numeric columns + Addicted_Score for correlation
corr_features = ['Age', 'Avg_Daily_Usage_Hours', 'Sleep_Hours_Per_Night',
                 'Mental_Health_Score', 'Addicted_Score']
corr_df = df[corr_features].dropna()   # drop rows with missing for correlation
plt.figure(figsize=(8,6))
sns.heatmap(corr_df.corr(), annot=True, cmap='coolwarm', fmt='.2f')
plt.title('Correlation Matrix of Numeric Features and Addicted Score')
plt.tight_layout()
plt.savefig('plots/04_correlation_heatmap.png')
plt.show()

print("EDA complete. All plots saved in 'plots/' folder.")