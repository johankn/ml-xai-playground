import shap
import matplotlib.pyplot as plt

def explain_model(model, X_test):
    explainer = shap.TreeExplainer(model, X_test)
    shap_values = explainer(X_test)

    # Global feature importance
    shap.summary_plot(shap_values[:,:,1], X_test, show=False)
    plt.savefig("shap_summary.png")

