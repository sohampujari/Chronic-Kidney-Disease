import joblib
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

def evaluate_model(../Data/kidney_disease_processed.csv):
    df = pd.read_csv(../Data/kidney_disease_processed.csv)
    X = df.drop(columns=['class'])
    y = df['class']

    model = joblib.load("model.pkl")
    y_pred = model.predict(X)

    accuracy = accuracy_score(y, y_pred)
    with open("results/accuracy_report.txt", "w") as f:
        f.write(f"Model Accuracy: {accuracy:.4f}\n")

    report = classification_report(y, y_pred)
    with open("results/classification_report.txt", "w") as f:
        f.write(report)

    cm = confusion_matrix(y, y_pred)
    plt.figure(figsize=(6,5))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['No CKD', 'CKD'], yticklabels=['No CKD', 'CKD'])
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.title("Confusion Matrix")
    plt.savefig("results/confusion_matrix.png")
    plt.close()

if __name__ == "__main__":
    evaluate_model("../Data/kidney_disease_processed.csv")

