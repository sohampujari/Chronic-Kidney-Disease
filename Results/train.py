import joblib
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier

def train_model(data_path):
    df = pd.read_csv(data_path)
    X = df.drop(columns=['class'])
    y = df['class']

    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X, y)

    joblib.dump(model, "model.pkl")

    importance = model.feature_importances_
    feature_names = X.columns
    plt.figure(figsize=(8,6))
    plt.barh(feature_names, importance, color='teal')
    plt.xlabel("Feature Importance")
    plt.ylabel("Features")
    plt.title("Feature Importance in CKD Prediction")
    plt.savefig("results/feature_importance.png")
    plt.close()

if __name__ == "__main__":
    train_model("../Data/kidney_disease_processed.csv")
