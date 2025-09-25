import numpy as np
from sklearn.model_selection import train_test_split
import pandas as pd
from sklearn.preprocessing import OneHotEncoder
from pipeline.loader import load_data

def preprocess(train_df, target_column, case="randomforest", sample_size=2000):
    drop_cols = ["id", "CustomerId", "Surname"]
    train_df = train_df.dropna()
    train_df = train_df.drop(columns=drop_cols)

    # Binary encode Gender
    train_df["Gender"] = train_df["Gender"].apply(lambda x: 1 if x == 'Male' else 0) 

    # OneHotEncode Geography
    ohe = OneHotEncoder(sparse_output=False, drop=None, handle_unknown='ignore')
    geography_encoded = ohe.fit_transform(train_df[["Geography"]])
    geography_df = pd.DataFrame(
        geography_encoded, 
        columns=ohe.get_feature_names_out(["Geography"]),
        index=train_df.index
    )
    train_df = train_df.drop(columns=["Geography"]).join(geography_df) 

    # Split X and y
    X = train_df.drop(columns=[target_column])
    y = train_df[target_column]

    # Sample (optional)
    if sample_size and len(X) > sample_size:
        X = X.head(sample_size)
        y = y.head(sample_size)

    # Case handling
    case = case.lower()
    if case in ["randomforest", "xgboost"]:
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )
    elif case == "linear":
        # Linear regression i sklearn forventer numpy
        X_train, X_test, y_train, y_test = train_test_split(
            X.to_numpy(dtype=np.float32),
            y.to_numpy(dtype=np.float32),
            test_size=0.2,
            random_state=42
        )
    elif case == "tsetlin":
        # Tsetlin-maskinen krever heltalls numpy arrays
        X_train, X_test, y_train, y_test = train_test_split(
            X.to_numpy(dtype=np.uint32),
            y.to_numpy(dtype=np.uint32),
            test_size=0.2,
            random_state=42
        )

    else:
        raise ValueError(f"Ukjent case: {case}")

    return X_train, X_test, y_train, y_test

