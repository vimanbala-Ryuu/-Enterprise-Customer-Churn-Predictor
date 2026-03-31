import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
import joblib
import json
import os
import re

def clean_col(c):
    return re.sub(r'[^a-zA-Z0-9]', '', str(c)).lower()

def safe_numeric(col):
    # Removes commas and dollar signs before converting to a number
    return pd.to_numeric(col.astype(str).str.replace(r'[,\$]', '', regex=True), errors='coerce').fillna(0)

def train_model():
    print("Starting DeepRetain AI training process...")
    try:
        df = pd.read_csv('customer_churn_dataset.csv', encoding='utf-8-sig')
        df.columns = [str(c).strip() for c in df.columns]
        df = df.dropna(subset=['Churn'])
        y = df['Churn'].astype(int)
        
        features = ['Tenure', 'Usage Frequency', 'Support Calls', 'Payment Delay', 'Total Spend', 'Last Interaction']
        
        # Safely map columns
        col_map = {clean_col(c): c for c in df.columns}
        X_data = {}
        for f in features:
            clean_f = clean_col(f)
            if clean_f in col_map:
                X_data[f] = safe_numeric(df[col_map[clean_f]])
            else:
                X_data[f] = [0.0] * len(df)
                
        X = pd.DataFrame(X_data)
        X = X[features] # Enforce strict column order
        
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        
        print("Training Random Forest. This will provide softer, more realistic probabilities...")
        # THE FIX: Swapped to RandomForest with depth limits to prevent absolute 100% certainty outputs
        model = RandomForestClassifier(n_estimators=100, max_depth=6, min_samples_leaf=4, random_state=42)
        model.fit(X_scaled, y)
        
        accuracy = round(model.score(X_scaled, y) * 100, 1)
        total_customers = len(df)
        
        os.makedirs('models', exist_ok=True)
        joblib.dump(model, 'models/churn_model.pkl')
        joblib.dump(scaler, 'models/scaler.pkl')
        
        metrics = {"model_accuracy": accuracy, "total_customers": total_customers, "features": features}
        with open('models/metrics.json', 'w') as f:
            json.dump(metrics, f)
            
        print(f"\n✅ Success! Model trained with {accuracy}% accuracy on {total_customers} records.")
    except Exception as e:
        print(f"\n❌ Error during training: {e}")

if __name__ == "__main__":
    train_model()