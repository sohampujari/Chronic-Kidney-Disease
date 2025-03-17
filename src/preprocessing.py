import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import LabelEncoder

def preprocess_data(../Data/kidney_disease.csv):
    df = pd.read_csv(../Data/kidney_disease.csv)
    
df = preprocess_data(../Data/kidney_disease.csv)
save_data(df, ../Data/kidney_disease.csv)

    imputer = SimpleImputer(strategy="mean")
    df.fillna(df.mean(), inplace=True)

    label_encoders = {}
    for column in df.select_dtypes(include=['object']).columns:
        label_encoders[column] = LabelEncoder()
        df[column] = label_encoders[column].fit_transform(df[column])

    return df
