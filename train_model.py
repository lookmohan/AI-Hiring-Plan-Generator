import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from xgboost import XGBClassifier
import joblib
import warnings
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

# Load dataset
df = pd.read_csv("ai_hiring_plan_dataset_700_rows.csv")

# ✅ Use relevant input features from your dataset
features = [
    'Industry', 'Company_Size', 'Tech_Stack', 'Hiring_Budget', 'Role',
    'Seniority_Level', 'Skills_Required', 'Years_Experience',
    'Project_Type', 'Project_Timeline', 'Expected_Deliverables',
    'Current_Team', 'Collaboration_Model', 'Diversity_Goals',
    'Number_of_Hires', 'Hiring_Method'
]

# ✅ For testing, we'll create 3 dummy target columns (you can replace with real targets if available)
df['Recommended_Title'] = df['Role']
df['Estimated_Cost'] = df['Hiring_Budget']
df['Estimated_Time'] = df['Project_Timeline']

# Create feature input by joining all columns into one string and using its length as numeric input
X = df[features].astype(str).agg(' '.join, axis=1).apply(len).to_frame()

targets = ['Recommended_Title', 'Estimated_Cost', 'Estimated_Time']
model_dict = {}

for target in targets:
    y = df[target].astype(str)

    # Label encode target
    le = LabelEncoder()
    y_encoded = le.fit_transform(y)

    # Train Random Forest
    rf = RandomForestClassifier(n_estimators=100, random_state=42)
    rf.fit(X, y_encoded)

    # Train XGBoost
    xgb = XGBClassifier(use_label_encoder=False, eval_metric='mlogloss', random_state=42)
    xgb.fit(X, y_encoded)

    model_dict[target] = {
        "RandomForest": rf,
        "XGBoost": xgb,
        "LabelEncoder": le
    }

# Save to joblib
joblib.dump(model_dict, "ai_hiring_models.joblib")
print("✅ Model training complete. Saved as ai_hiring_models.joblib")
