import pandas as pd
from sklearn.preprocessing import StandardScaler

def preprocess_tabular(csv_path="../data/raw/data.csv"):
    data = pd.read_csv(csv_path)
    
    if 'id' in data.columns:
        data = data.drop('id', axis=1)
    
    data['diagnosis'] = data['diagnosis'].map({'M':1, 'B':0})
    
    X = data.drop('diagnosis', axis=1)
    y = data['diagnosis']
    
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    return X_scaled, y.values
