import joblib
import pandas as pd
from sklearn.metrics import classification_report, confusion_matrix

def evaluate_model(data_path):
    df = pd.read_csv(data_path)
    X = df.drop(columns=['class'])
    y = df['class']

    model = joblib.load("model.pkl")
    y_pred = model.predict(X)

    print("Classification Report:")
    print(classification_report(y, y_pred))

    print("Confusion Matrix:")
    print(confusion_matrix(y, y_pred))

if __name__ == "__main__":
    evaluate_model("../data/processed/ckd_processed.csv")
