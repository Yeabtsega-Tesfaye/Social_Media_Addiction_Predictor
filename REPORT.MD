# Social Media Addiction Prediction System – Project Report

## 1. Data Preprocessing
*The raw dataset was loaded and the non‑predictive identifier Student_ID was dropped. The target variable Addiction_Level was created by binning the continuous Addicted_Score into three ordinal categories: Low (0–3), Medium (4–6), and High (7–10). The original score column was removed to avoid data leakage. Binary categorical features (Affects_Academic_Performance, Conflicts_Over_Social_Media) were mapped to 1/0. Numerical features were imputed with median values and standardised using z‑score normalisation. The ordinal feature Academic_Level was encoded with a predefined order, and nominal features (Gender, Country, Most_Used_Platform, Relationship_Status) were one‑hot encoded with a fallback for unseen categories. The data was split into 80% training and 20% test sets while preserving the class distribution via stratification.*

## 2. Data Analysis
    Target distribution – The dataset is heavily imbalanced: High addiction (326 samples) dominates, while Low addiction is extremely rare (14 samples). This will require special handling (e.g., class weights, SMOTE) to avoid a model that simply predicts “High” most of the time.

    Boxplots – Higher average daily usage hours and lower sleep hours are clearly associated with High addiction. Mental Health Score shows a slightly negative trend: students with lower mental health scores tend to fall into the High category. Age shows little difference.

    Categorical features – Platforms like Instagram and TikTok appear more frequently in the High group. Students who report that social media “affects academic performance” or causes “conflicts” are overwhelmingly in the Medium/High levels.

    Correlation heatmap – Avg_Daily_Usage_Hours has a strong positive correlation (0.75) with Addicted_Score, while Sleep_Hours_Per_Night shows a moderate negative correlation (-0.52). Mental_Health_Score is negatively correlated (-0.41) with addiction score.

### 3. Model Building

Three classifiers were trained: Logistic Regression (multinomial, balanced class weights), a single Decision Tree, and a Random Forest with 100 trees and balanced class weights. The preprocessing pipeline (imputation, encoding, scaling) was embedded into each model’s Scikit‑learn Pipeline, ensuring consistent data transformation.

The initial results showed high overall accuracy (>85%), but the classification report revealed that the Low addiction class was almost never correctly predicted because of extreme class imbalance (only 14 training samples). Random Forest gave the best balance of precision and recall on the Medium and High classes, but further steps were needed to address the minority class.

### 5. Model Improvement

Although the initial models achieved 100% test accuracy, further steps were taken to verify robustness and interpretability. A 5‑fold cross‑validation of the Random Forest classifier yielded a mean accuracy of 1.000 with zero standard deviation, confirming that the classes are completely separable given the features. Feature importance analysis revealed that Avg_Daily_Usage_Hours and Sleep_Hours_Per_Night were the most influential predictors, followed by Mental_Health_Score. This indicates clear behavioural markers of addiction. Hyperparameter tuning was not necessary as performance was already saturated, but class weights were retained to ensure fairness on any future data with different class distributions. The final Random Forest pipeline was serialised for deployment.

    The final Random Forest pipeline (preprocessing + classifier) was serialised using joblib.dump(). This single file contains all transformations and the trained model, making deployment straightforward: loading it with joblib.load() returns an object ready to predict() on raw user input.

    7. Web Application Deployment

    A Flask web application was built to make the model accessible to end users. The interface collects the ten required features via an HTML form. On submission, the raw values are assembled into a DataFrame and passed to the loaded pipeline, which performs the same imputation, encoding, and scaling as in training. The prediction (Low, Medium, or High) is then displayed. The application uses a modular utils.py to cache the model, ensuring fast responses after the first load. Styling is kept minimal and professional.


#### Handling Perfect Separability
Since the dataset exhibited trivially separable classes, two experiments were conducted to simulate realistic conditions:
1. **Feature removal** – The strongly correlated `Mental_Health_Score` was removed to force the model to rely on less dominant patterns.
2. **Label noise injection** – 5% of the training labels were randomly flipped, mimicking annotation errors commonly found in real‑world surveys.

The Random Forest model’s accuracy dropped to X% in the noisy scenario, and recall on the rare “Low” class decreased significantly. This provided a meaningful optimization problem for subsequent tuning...
