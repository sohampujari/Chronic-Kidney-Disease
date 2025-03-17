import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import LabelEncoder

def preprocess_data(data_path):
    df = pd.read_csv(data_path)
    
    # Handling missing values
    imputer = SimpleImputer(strategy="mean")
    df.fillna(df.mean(), inplace=True)

    # Encoding categorical features
    label_encoders = {}
    for column in df.select_dtypes(include=['object']).columns:
        label_encoders[column] = LabelEncoder()
        df[column] = label_encoders[column].fit_transform(df[column])

    return df
