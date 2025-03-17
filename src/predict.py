import joblib
import pandas as pd

def predict_new_data(new_data):
    model = joblib.load("model.pkl")
    df = pd.DataFrame(new_data)
    predictions = model.predict(df)
    return predictions

if __name__ == "__main__":
    sample_data = [{"age": 50, "bp": 80, "sg": 1.02, "al": 1, "su": 0, "hemo": 13.5}]
    print("Predictions:", predict_new_data(sample_data))
