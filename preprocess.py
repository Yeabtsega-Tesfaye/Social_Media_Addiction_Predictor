import pandas as pd
from pandas.api.types import CategoricalDtype
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OrdinalEncoder, OneHotEncoder
import joblib
import os

# Load the dataset
filepath = 'data/dataset.csv'
df = pd.read_csv(filepath)

# 1. Drop Student_ID (if present)
df.drop('Student_ID', axis=1, inplace=True, errors='ignore')

# 2. Create Addiction_Level from Addicted_Score
bins = [0, 3, 6, 10]
labels = ['Low', 'Medium', 'High']
df['Addiction_Level'] = pd.cut(df['Addicted_Score'], bins=bins, labels=labels,
                               include_lowest=True, right=True)

# 3. Drop the original score
df.drop('Addicted_Score', axis=1, inplace=True)

# 4. Separate features and target
X = df.drop('Addiction_Level', axis=1)
y = df['Addiction_Level']
# Keep y as a pandas Series, but enforce the category order
cat_type = CategoricalDtype(categories=['Low', 'Medium', 'High'], ordered=False)
y = y.astype(cat_type)   # now y is a Series with fixed category order

# 5. Manually encode binary columns (map Yes/No -> 1/0)
binary_cols = ['Affects_Academic_Performance', 'Conflicts_Over_Social_Media']
for col in binary_cols:
    X[col] = X[col].map({'Yes': 1, 'No': 0}).astype(float)

# ---- NEW: Drop columns that are 100% missing ----
# If a column became all NaN after mapping, drop it
for col in binary_cols:
    if X[col].isna().all():
        print(f"Dropping entirely missing column: {col}")
        X.drop(col, axis=1, inplace=True)
        binary_cols.remove(col)   # remove from the list so it won't be included later

# 6. Define column groups (after possible drop)
numeric_cols = ['Age', 'Avg_Daily_Usage_Hours', 'Sleep_Hours_Per_Night', 'Mental_Health_Score'] + binary_cols
ordinal_cols = ['Academic_Level']
nominal_cols = ['Gender', 'Country', 'Most_Used_Platform', 'Relationship_Status']

# 7. Define the order for Academic_Level (adjust to your actual categories)
academic_order = ['High School', 'Undergraduate', 'Graduate']

# 8. Build the ColumnTransformer
numeric_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='median')),
    ('scaler', StandardScaler())
])

ordinal_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='most_frequent')),
    ('ordinal', OrdinalEncoder(categories=[academic_order]))
])

nominal_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='constant', fill_value='missing')),
    ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
])

preprocessor = ColumnTransformer(transformers=[
    ('num', numeric_transformer, numeric_cols),
    ('ord', ordinal_transformer, ordinal_cols),
    ('nom', nominal_transformer, nominal_cols)
])

# ---- NEW: Force target category order ----
y = pd.Categorical(y, categories=['Low', 'Medium', 'High'], ordered=False)

# 9. Split into train/test (stratified)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# Force them back to Series with the same categorical type
y_train = pd.Series(y_train, name='Addiction_Level', dtype=cat_type)
y_test  = pd.Series(y_test, name='Addiction_Level', dtype=cat_type)

print("Training set shape:", X_train.shape)
print("Test set shape:", X_test.shape)
print("Class distribution in train:\n", y_train.value_counts())
print("Preprocessor ready.")

os.makedirs('data', exist_ok=True)
os.makedirs('models', exist_ok=True)

# Save the preprocessor
joblib.dump(preprocessor, 'models/preprocessor.pkl')

# Save the train/test sets
X_train.to_csv('data/X_train.csv', index=False)
X_test.to_csv('data/X_test.csv', index=False)
y_train.to_csv('data/y_train.csv', index=False, header=True)
y_test.to_csv('data/y_test.csv', index=False, header=True)

print("Preprocessor and data saved successfully.")