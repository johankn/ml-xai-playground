import shap
import numpy as np
from sklearn.ensemble import RandomForestClassifier
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

X = np.random.rand(10, 5)
y = np.random.randint(0, 2, size=10)
model = RandomForestClassifier(n_estimators=10, random_state=42)
model.fit(X, y)

explainer = shap.Explainer(model, X)
shap_values = explainer(X)
print("SHAP works")

# Summary plot
shap.summary_plot(shap_values, X, feature_names=[f"f{i}" for i in range(X.shape[1])], show=False)
plt.savefig("shap_summary.png")
print("Summary plot saved as shap_summary.png")

