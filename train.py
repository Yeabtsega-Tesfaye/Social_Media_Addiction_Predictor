import pandas as pd
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
import os
import numpy as np

from pandas.api.types import CategoricalDtype
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.model_selection import cross_val_score

# Create folders for plots if needed
os.makedirs('plots', exist_ok=True)

# ----- 1. Load data and preprocessor -----
X_train = pd.read_csv('data/X_train.csv')
X_test  = pd.read_csv('data/X_test.csv')
y_train_series = pd.read_csv('data/y_train.csv').squeeze()
y_test_series  = pd.read_csv('data/y_test.csv').squeeze()

# Define the exact categorical type used in preprocessing
cat_type = CategoricalDtype(categories=['Low', 'Medium', 'High'], ordered=False)
y_train = y_train_series.astype(cat_type)
y_test  = y_test_series.astype(cat_type)

preprocessor = joblib.load('models/preprocessor.pkl')

# ----- 2. Define models -----
models = {
    'Logistic Regression': Pipeline([
        ('preprocessor', preprocessor),
        ('classifier', LogisticRegression(
            solver='lbfgs',
            max_iter=1000,
            class_weight='balanced',
            random_state=42
        ))
    ]),
    'Decision Tree': Pipeline([
        ('preprocessor', preprocessor),
        ('classifier', DecisionTreeClassifier(random_state=42))
    ]),
    'Random Forest': Pipeline([
        ('preprocessor', preprocessor),
        ('classifier', RandomForestClassifier(
            n_estimators=100,
            random_state=42,
            class_weight='balanced'
        ))
    ])
}

# ----- 3. Train and evaluate each model -----
for name, pipe in models.items():
    print(f"\n{'='*40}")
    print(f"Training {name}...")
    pipe.fit(X_train, y_train)
    y_pred = pipe.predict(X_test)

    acc = accuracy_score(y_test, y_pred)
    cm = confusion_matrix(y_test, y_pred, labels=pipe.classes_)
    report = classification_report(y_test, y_pred, target_names=pipe.classes_)

    print(f"Accuracy: {acc:.4f}")
    print("Classification Report:")
    print(report)

    # Plot confusion matrix (using pipe.classes_ for correct order)
    plt.figure(figsize=(6,5))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=pipe.classes_, yticklabels=pipe.classes_)
    plt.title(f'Confusion Matrix – {name}')
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.tight_layout()
    plt.savefig(f'plots/confusion_matrix_{name.lower().replace(" ", "_")}.png')
    plt.close()

# ----- 4. Improvement: Cross-validation on Random Forest -----
print("\n" + "="*40)
print("5‑fold Cross‑Validation (Random Forest)")
rf_pipe = models['Random Forest']
cv_scores = cross_val_score(rf_pipe, X_train, y_train, cv=5, scoring='accuracy')
print(f"CV accuracy (mean ± std): {cv_scores.mean():.4f} ± {cv_scores.std():.4f}")

# ----- 5. Feature Importance -----
# Fit once more to get feature names after transformation
rf_pipe.fit(X_train, y_train)
feature_names = rf_pipe.named_steps['preprocessor'].get_feature_names_out()
importances = rf_pipe.named_steps['classifier'].feature_importances_
feat_imp = pd.Series(importances, index=feature_names).sort_values(ascending=False)
print("\nTop 10 Most Important Features:")
print(feat_imp.head(10))

# Plot feature importances
plt.figure(figsize=(10, 6))
feat_imp.head(10).plot(kind='bar')
plt.title('Random Forest Feature Importances')
plt.tight_layout()
plt.savefig('plots/feature_importances.png')
plt.close()

# ----- 6. Save the best model -----
joblib.dump(rf_pipe, 'models/social_media_addiction_model.pkl')
print("\nBest model (Random Forest) saved as 'models/social_media_addiction_model.pkl'")