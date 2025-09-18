from sklearn.ensemble import RandomForestClassifier
import pandas as pd
import matplotlib.pyplot as plt

def train_model(X_train, y_train):
    model = RandomForestClassifier(class_weight='balanced', n_estimators=100, random_state=42)
    model.fit(X_train, y_train)

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

