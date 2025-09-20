import os
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
from sklearn.model_selection import StratifiedKFold
from sklearn.linear_model import LogisticRegression, RidgeClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, HistGradientBoostingClassifier, AdaBoostClassifier
from xgboost import XGBClassifier
from sklearn.metrics import (
    accuracy_score, recall_score, precision_recall_curve,
    f1_score, roc_curve, auc, confusion_matrix
)
from imblearn.over_sampling import RandomOverSampler
from tqdm import tqdm
import warnings

warnings.filterwarnings("ignore")

# ====== Paths ======
base_dir = "/mnt/29T_HardDisk/huzihan_private/广东省第二人民医院/杨主任论文"
model_dir = os.path.join(base_dir, "非标准化模型")
viz_dir   = os.path.join(base_dir, "非标准化可视化图片")
os.makedirs(model_dir, exist_ok=True)
os.makedirs(viz_dir, exist_ok=True)

# ====== Load data ======
X = pd.read_csv(os.path.join(base_dir, "X_con.csv"))
y = pd.read_csv(os.path.join(base_dir, "y_con.csv")).squeeze("columns")

# ====== Specificity helper ======
def specificity_score(y_true, y_pred):
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    return tn / (tn + fp)

# ====== Define models ======
models = {
    "LogisticRegression": LogisticRegression(max_iter=10000, random_state=42),
    "RidgeClassifier":      RidgeClassifier(),
    "DecisionTree":         DecisionTreeClassifier(criterion='entropy', min_samples_leaf=5, random_state=42),
    "RandomForest":         RandomForestClassifier(n_estimators=500, min_samples_leaf=5, max_features='sqrt', random_state=42),
    "XGBoost":              XGBClassifier(subsample=0.6, reg_lambda=1, n_estimators=300, max_depth=4,
                                           learning_rate=0.01, gamma=1, colsample_bytree=1.0,
                                           use_label_encoder=False, eval_metric='logloss', random_state=42),
    "HistGradientBoosting": HistGradientBoostingClassifier(max_leaf_nodes=15, learning_rate=0.05,
                                                           l2_regularization=0.5, random_state=42),
    "AdaBoost_DT":          AdaBoostClassifier(DecisionTreeClassifier(max_depth=1),
                                               n_estimators=10, random_state=42)
}

# ====== Cross-validation setup ======
cv        = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)
threshold = 0.5
results   = []

# ====== Run CV for each model ======
for name, model in tqdm(models.items(), desc="Training models"):
    accs, sens, specs, f1s, aucs = [], [], [], [], []
    for train_idx, test_idx in cv.split(X, y):
        X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
        y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]

        # balance training set
        X_res, y_res = RandomOverSampler(random_state=42).fit_resample(X_train, y_train)
        model.fit(X_res, y_res)

        # get predicted probabilities
        if hasattr(model, "predict_proba"):
            y_proba = model.predict_proba(X_test)[:, 1]
        else:
            dfun = model.decision_function(X_test)
            y_proba = (dfun - dfun.min()) / (dfun.max() - dfun.min())

        # compute metrics at fixed threshold
        y_pred = (y_proba >= threshold).astype(int)
        accs.append(   accuracy_score(y_test, y_pred))
        sens.append(  recall_score(y_test, y_pred))
        specs.append(specificity_score(y_test, y_pred))
        f1s.append(   f1_score(y_test, y_pred))

        # AUC
        fpr, tpr, _ = roc_curve(y_test, y_proba)
        aucs.append(auc(fpr, tpr))

    # aggregate
    results.append({
        'Model':            name,
        'Accuracy (0.5)':   np.mean(accs),
        'Sensitivity (0.5)':np.mean(sens),
        'Specificity (0.5)':np.mean(specs),
        'F1 Score (0.5)':   np.mean(f1s),
        'AUC (0.5)':        np.mean(aucs)
    })

    # save trained model
    joblib.dump(model, os.path.join(model_dir, f"{name}_model.pkl"))

# ====== Save evaluation results ======
results_df = pd.DataFrame(results).set_index('Model').round(4)
results_df.to_csv(os.path.join(model_dir, "evaluation_results.csv"))

# ====== Plot AUC comparison ======
plt.figure(figsize=(8, 5))
results_df['AUC (0.5)'].sort_values().plot.bar()
plt.ylabel("AUC (threshold=0.5)")
plt.title("Model AUC Comparison (Non-standardized)")
plt.tight_layout()
plt.savefig(os.path.join(viz_dir, "AUC_comparison.png"))
plt.close()

# (Optional) you could similarly plot other metrics, e.g. Accuracy, F1, etc.
print("Done. Models, metrics, and plots have been saved.")
