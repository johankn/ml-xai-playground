from sklearn.model_selection import train_test_split
import pandas as pd
from sklearn.preprocessing import OneHotEncoder
from pipeline.loader import load_data

def preprocess(train_df, target_column):
    drop_cols = ["id", "CustomerId", "Surname"]
    train_df = train_df.dropna()
    train_df = train_df.drop(columns=drop_cols)
    train_df["Gender"] = train_df["Gender"].apply(lambda x: 1 if x == 'Male' else 0) 
    ohe = OneHotEncoder(sparse_output=False, drop=None, handle_unknown='ignore')
    geography_encoded = ohe.fit_transform(train_df[["Geography"]])

    # Gjør om til DataFrame med kolonnenavn
    geography_df = pd.DataFrame(
        geography_encoded, 
        columns=ohe.get_feature_names_out(["Geography"])
    )

    # Slå sammen med originalt datasett
    train_df = train_df.drop(columns=["Geography"]).join(geography_df) 

    print("Preprocessing complete. Here are the first few rows:", train_df.head(10))

    X = train_df.drop(columns=[target_column])
    y = train_df[target_column]

    X_sample = X.head(500)
    y_sample = y.head(500)


    X_train, X_test, y_train, y_test = train_test_split(X_sample, y_sample, test_size=0.2, random_state=42)

    return X_train, X_test, y_train, y_test


preprocess(load_data("data/train.csv"), target_column="Exited")
