import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
import joblib

# 1. Load your dataset
df = pd.read_csv('data.csv')

# 2. Clean and Preprocess
to_drop = ['EmployeeCount', 'Over18', 'StandardHours', 'EmployeeNumber']
df_clean = df.drop(columns=to_drop)

categorical_cols = df_clean.select_dtypes(include=['object']).columns.tolist()
categorical_cols.remove('Attrition')

mappings = {}
for col in categorical_cols:
    le = LabelEncoder()
    df_clean[col] = le.fit_transform(df_clean[col])
    mappings[col] = dict(zip(le.classes_, range(len(le.classes_))))

df_clean['Attrition'] = df_clean['Attrition'].map({'Yes': 1, 'No': 0})

# 3. Train Model
X = df_clean.drop('Attrition', axis=1)
y = df_clean['Attrition']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

rf_full = RandomForestClassifier(n_estimators=100, random_state=42)
rf_full.fit(X_train, y_train)

# 4. Save the artifacts required by the app
joblib.dump(rf_full, 'model_full.pkl')
joblib.dump(mappings, 'mappings.pkl')
joblib.dump(X.columns.tolist(), 'feature_names.pkl')

print("Model files generated successfully!")