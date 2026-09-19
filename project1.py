import numpy as np
import pandas as pd
from sklearn.impute import KNNImputer
import pandera as pa

# 1. Sample Raw Dataset
np.random.seed(42)
data = {
    'income': [50000, 55000, np.nan, 60000, 1200000, 52000, 58000, np.nan, 51000, 54000],
    'age': [25, np.nan, 28, 35, 40, 22, np.nan, 30, 29, 31],
    'score': [80, 85, 90, np.nan, 95, 70, 65, 88, 92, 78],
    'city': ['London', 'Paris', 'Tokyo', 'London', 'Paris', 'Tokyo', 'London', 'Paris', 'Tokyo', 'London'],
    'target': [1, 0, 1, 0, 1, 0, 1, 0, 1, 0]
}
df = pd.DataFrame(data)

# 2. MODULE 1: INPUT - Missing Data Imputation & Outlier Handling
for col in df.select_dtypes(include=[np.number]).columns:
    null_ratio = df[col].isnull().mean()
    if null_ratio < 0.05:
        df = df.dropna(subset=[col])
    elif 0.05 <= null_ratio <= 0.20:
        df[col] = df[col].fillna(df[col].median())
    else:
        imputer = KNNImputer(n_neighbors=3)
        df[[col]] = imputer.fit_transform(df[[col]])

# Outlier Neutralization via IQR Winsorization
Q1 = df['income'].quantile(0.25)
Q3 = df['income'].quantile(0.75)
IQR = Q3 - Q1
df['income'] = np.clip(df['income'], Q1 - 1.5 * IQR, Q3 + 1.5 * IQR)

# 3. MODULE 2: PROCESS - Encoding & Feature Engineering
df = pd.get_dummies(df, columns=['city'], drop_first=False)

# Engineering 3 New Predictive Features
df['income_per_age'] = df['income'] / df['age']
df['score_age_ratio'] = df['score'] / df['age']
df['income_score_prod'] = df['income'] * df['score']

# 4. MODULE 3: OUTPUT - Pandera Schema Validation
schema = pa.DataFrameSchema({
    "income": pa.Column(float, pa.Check.ge(0)),
    "age": pa.Column(float, pa.Check.in_range(18, 100)),
    "target": pa.Column(int, pa.Check.isin([0, 1]))
}, coerce=True)

validated_df = schema.validate(df, lazy=True)

print("\n--- PIPELINE EXECUTED SUCCESSFULLY ---")
print("\nCleaned & Engineered Dataset Output:")
print(validated_df.head())