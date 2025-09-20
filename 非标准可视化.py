import os
import joblib
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import shap
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    confusion_matrix, ConfusionMatrixDisplay,
    roc_curve, auc
)
from imblearn.over_sampling import RandomOverSampler

# ====== Paths ======
base_dir   = "/mnt/29T_HardDisk/huzihan_private/广东省第二人民医院/杨主任论文"
data_X     = os.path.join(base_dir, "X_con.csv")
data_y     = os.path.join(base_dir, "y_con.csv")
orig_model = os.path.join(base_dir, "models3", "XGBoost_model.pkl")
model_dir  = os.path.join(base_dir, "非标准化模型")
viz_dir    = os.path.join(base_dir, "非标准化可视化图片")

os.makedirs(model_dir, exist_ok=True)
os.makedirs(viz_dir, exist_ok=True)

# ====== 1. Load data ======
X = pd.read_csv(data_X)
y = pd.read_csv(data_y).squeeze("columns")

# ====== 2. Load, retrain & save XGBoost ======
xgb = joblib.load(orig_model)
xgb.fit(X, y)

# save the retrained model
retrained_model_path = os.path.join(model_dir, "XGBoost_model_nonstd.pkl")
joblib.dump(xgb, retrained_model_path)

# ====== 3. Feature importance (top 80%) ======
importances = xgb.feature_importances_
idxs        = np.argsort(importances)[::-1]
feats       = X.columns[idxs]
vals        = importances[idxs]
cum_vals    = np.cumsum(vals)
cutoff      = np.searchsorted(cum_vals, 0.8) + 1
top_feats   = feats[:cutoff]
top_vals    = vals[:cutoff]

plt.figure(figsize=(8, max(4, 0.3 * len(top_feats))))
plt.barh(top_feats[::-1], top_vals[::-1])
plt.xlabel("Feature Importance")
plt.title("XGBoost Feature Importance (Top 80%)")
plt.tight_layout()
plt.savefig(os.path.join(viz_dir, "FeatureImportance80_XGBoost.png"))
plt.close()

# ====== 4. SHAP explanation ======
explainer = shap.TreeExplainer(xgb, X, model_output="probability")
shap_exp  = explainer(X, check_additivity=False)
if shap_exp.values.ndim == 3:
    sv_all   = shap_exp.values[:, :, 1]
    base_val = explainer.expected_value[1]
else:
    sv_all   = shap_exp.values
    base_val = explainer.expected_value

# 4.1 Summary plot
plt.figure(figsize=(10, 6))
shap.summary_plot(sv_all, X, show=False)
plt.title("SHAP Summary - XGBoost")
plt.tight_layout()
plt.savefig(os.path.join(viz_dir, "SHAP_Summary_XGBoost.png"))
plt.close()

# 4.2 Decision plot
plt.figure(figsize=(10, 6))
shap.decision_plot(
    base_value=base_val,
    shap_values=sv_all,
    features=X,
    feature_names=X.columns.tolist(),
    ignore_warnings=True,
    show=False
)
plt.title("SHAP Decision Plot - XGBoost")
plt.tight_layout()
plt.savefig(os.path.join(viz_dir, "SHAP_Decision_XGBoost.png"))
plt.close()

# 4.3 Waterfall for first sample with all-positive core SHAP
core_feats = ["age", "TyG_BMI_index", "hemoglobin_g_per_dL", "TyG_waist_height_index"]
core_idx   = [X.columns.get_loc(f) for f in core_feats]
mask_pos   = (sv_all[:, core_idx] > 0).all(axis=1)
candidates = np.where(mask_pos)[0]
if candidates.size:
    i = candidates[0]
    wf = shap.Explanation(
        values        = sv_all[i],
        base_values   = base_val,
        data          = X.iloc[i].values,
        feature_names = X.columns.tolist()
    )
    plt.figure(figsize=(8, 5))
    shap.waterfall_plot(wf, show=False)
    plt.title(f"SHAP Waterfall Sample{i} - XGBoost")
    plt.tight_layout()
    plt.savefig(os.path.join(viz_dir, f"SHAP_Waterfall_Sample{i}_XGBoost.png"))
    plt.close()

# ====== 5. SHAP dependence for core pairs ======
import itertools, math

pairs = list(itertools.combinations(core_feats, 2))
rows, cols = math.ceil(len(pairs) / 3), 3
fig, axes = plt.subplots(rows, cols, figsize=(cols * 4, rows * 3), squeeze=False)
axes = axes.flatten()

for ax, (f1, f2) in zip(axes, pairs):
    shap.dependence_plot(f1, sv_all, X, interaction_index=f2, show=False, ax=ax)
    ax.set_title(f"{f1} vs {f2}", fontsize=9)

for ax in axes[len(pairs):]:
    ax.axis("off")

plt.tight_layout()
plt.savefig(os.path.join(viz_dir, "SHAP_Dependence_Core4Pairs_XGBoost.png"), dpi=300)
plt.close()

# ====== 6. Correlation heatmap top20 SHAP ======
mean_abs = np.mean(np.abs(sv_all), axis=0)
top20    = np.argsort(mean_abs)[::-1][:20]
corr     = X.iloc[:, top20].corr()

plt.figure(figsize=(10, 10))
im = plt.imshow(corr, cmap="coolwarm", vmin=-1, vmax=1)
plt.colorbar(im, fraction=0.046, pad=0.04)
labels = X.columns[top20]
plt.xticks(range(20), labels, rotation=90)
plt.yticks(range(20), labels)
plt.title("Correlation Heatmap of Top 20 SHAP Features")
plt.tight_layout()
plt.savefig(os.path.join(viz_dir, "CorrelationHeatmap_Top20_SHAP.png"))
plt.close()

# ====== 7. Confusion matrix on hold-out ======
X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)
X_res, y_res = RandomOverSampler(random_state=42).fit_resample(X_tr, y_tr)
xgb.fit(X_res, y_res)

y_proba = xgb.predict_proba(X_te)[:, 1]
y_pred  = (y_proba >= 0.5).astype(int)

cm = confusion_matrix(y_te, y_pred)
disp = ConfusionMatrixDisplay(cm, display_labels=["No LOH", "LOH"])
plt.figure(figsize=(5, 4))
disp.plot(cmap="Blues", values_format="d", ax=plt.gca())
plt.title("XGBoost Confusion Matrix")
plt.tight_layout()
plt.savefig(os.path.join(viz_dir, "ConfusionMatrix_XGBoost.png"))
plt.close()


