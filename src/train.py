import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

def train_model(../Data/kidney_disease_processed.csv):
    df = pd.read_csv(../Data/kidney_disease_processed.csv)
    X = df.drop(columns=['class']) 
    y = df['class']
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    
    print(f"Model Accuracy: {accuracy:.2f}")
    
    joblib.dump(model, "model.pkl")

if __name__ == "__main__":
    train_model("../Data/kidney_disease_processed.csv")
