import time
import shap
import matplotlib.pyplot as plt
from interpret import show

def explain_model(model, X_test, churn_index):
    # explainer = shap.TreeExplainer(model, X_test)
    # shap_values = explainer(X_test)
    ebm_global = model.explain_global()
    show(ebm_global)
    print("Dashboard launched — open the link above in your browser.")
    print("Press Ctrl+C to exit.")
    while True:
        time.sleep(5)

    # # Global feature importance
    # shap.summary_plot(shap_values[:,:,1], X_test, show=False, max_display=20)
    # plt.yticks(rotation=0, fontsize=9)
    # plt.savefig("diagrams/shap_summary.png")
    # plt.close()

    # # Local explanation for the first predicted churn instance
    # shap.plots.waterfall(shap_values[churn_index,:,1], show=False, max_display=20)
    # plt.gcf().set_size_inches(10, 12)   # make the figure larger (width x height in inches)
    # plt.tight_layout() 
    # plt.savefig("diagrams/shap_waterfall.png")
    # plt.close()

