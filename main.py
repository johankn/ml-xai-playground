import numpy as np
from pipeline.loader import load_data
from pipeline.preprocess import preprocess
from models.train import train_model
from explain.shap_explain import explain_model
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, precision_recall_curve, auc, accuracy_score

# -----------------------------
# 1. Load data
# -----------------------------
train_df = load_data("data/train.csv")
# test_df = load_data("data/test.csv")  # test.csv har ingen target

# -----------------------------
# 2. Preprocess
# -----------------------------
target_column = "Exited"
X_train, X_test, y_train, y_test = preprocess(train_df, target_column=target_column, case="ebm", sample_size=10000)

# -----------------------------
# 3. Train model
# -----------------------------
model = train_model(X_train, y_train, case="ebm")
y_pred = model.predict(X_test)
print("ypred", y_pred)

# find the indices of the predicted churn instances
churn_indices = np.where(y_pred==1)[0]
churn_index = churn_indices[2]
print(f"Churn index: {churn_index}")

# Accuracy and classification report
print(f"Accuracy: {accuracy_score(y_test, y_pred):.4f}")
print(classification_report(y_test, y_pred, digits=3))

# Extra metrics for imbalance
roc = roc_auc_score(y_test, model.predict_proba(X_test)[:,1])
print(f"ROC-AUC: {roc:.4f}")

# Precision-Recall AUC (more informative for imbalance)
precision, recall, _ = precision_recall_curve(y_test, model.predict_proba(X_test)[:,1])
pr_auc = auc(recall, precision)
print(f"PR-AUC: {pr_auc:.4f}")

print("Confusion matrix:")
print(confusion_matrix(y_test, y_pred))


# -----------------------------
# 4. Explain predictions
# -----------------------------
explain_model(model, X_test, churn_index)
