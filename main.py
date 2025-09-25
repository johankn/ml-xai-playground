import numpy as np
from sklearn.metrics import accuracy_score, classification_report
from pipeline.loader import load_data
from pipeline.preprocess import preprocess
from models.train import train_model
from explain.shap_explain import explain_model

# -----------------------------
# 1. Load data
# -----------------------------
train_df = load_data("data/train.csv")
test_df = load_data("data/test.csv")  # test.csv har ingen target

# -----------------------------
# 2. Preprocess
# -----------------------------
target_column = "Exited"
X_train, X_test, y_train, y_test = preprocess(train_df, target_column=target_column, case="randomforest", sample_size=1000)

# -----------------------------
# 3. Train model
# -----------------------------
model = train_model(X_train, y_train, case="randomforest")
y_pred = model.predict(X_test)
print("ypred", y_pred)

# find the indices of the predicted churn instances
churn_indices = np.where(y_pred==1)[0]
churn_index = churn_indices[2]
print(f"Churn index: {churn_index}")

# Accuracy and classification report
acc = accuracy_score(y_test, y_pred)
print(f"Accuracy randomforest: {acc:.4f}")
print(classification_report(y_test, y_pred))

# -----------------------------
# 4. Explain predictions
# -----------------------------
explain_model(model, X_test, churn_index)
