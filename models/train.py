from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
import xgboost as xgb
from pyTsetlinMachine.tm import MultiClassTsetlinMachine
import pandas as pd
import matplotlib.pyplot as plt
from interpret.glassbox import ExplainableBoostingClassifier


def train_model(X_train, y_train, case="randomforest"):
    case = case.lower()

    if case == "randomforest":
        model = RandomForestClassifier(
            class_weight="balanced",
            n_estimators=100,
            random_state=42
        )
        model.fit(X_train, y_train)

    elif case == "xgboost":
        model = xgb.XGBClassifier(
            n_estimators=200,
            learning_rate=0.1,
            max_depth=5,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=42,
            use_label_encoder=False,
            eval_metric="logloss"
        )
        model.fit(X_train, y_train)

    elif case == "ebm":
        model = ExplainableBoostingClassifier(random_state=42, interactions=5)
        model.fit(X_train, y_train)

    elif case == "linear":
        model = LogisticRegression(
            max_iter=5000,
            class_weight="balanced",
            random_state=42
        )
        model.fit(X_train, y_train)

    elif case == "tsetlin":
        model = MultiClassTsetlinMachine(number_of_clauses=1000, T=8000, s=1)
        model.fit(X_train, y_train, epochs=50)

    else:
        raise ValueError(f"Ukjent case: {case}")
    
    # feature_importances = model.feature_importances_
    # features_df = pd.DataFrame({'Feature': X_train.columns, 'Importance': feature_importances})
    # features_df = features_df.sort_values(by='Importance', ascending=False)
    # plt.figure(figsize=(10, 6))
    # plt.barh(features_df['Feature'], features_df['Importance'], color='skyblue')
    # plt.xlabel('Feature Importance')
    # plt.ylabel('Feature Name')
    # plt.title('Random Forest Feature Importance')
    # plt.gca().invert_yaxis() # To display the most important feature at the top
    # plt.savefig('feature_importance.png')

    return model

