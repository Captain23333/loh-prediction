import os
import joblib
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.inspection import PartialDependenceDisplay
from itertools import permutations

# ====== Paths ======
base_dir    = "/mnt/29T_HardDisk/huzihan_private/广东省第二人民医院/杨主任论文"
model_path  = os.path.join(base_dir, "非标准化模型", "XGBoost_model_nonstd.pkl")
X_vis_path  = os.path.join(base_dir, "X_con.csv")
output_dir  = os.path.join(base_dir, "非标准化可视化图片")
os.makedirs(output_dir, exist_ok=True)

# ====== Load data and model ======
X_vis   = pd.read_csv(X_vis_path)
xgb_full = joblib.load(model_path)

# ====== Features for PDP ======
pdp_features = [
    'age',
    'TyG_BMI_index',
    'hemoglobin_g_per_dL',
    'TyG_waist_height_index'
]

# ====== 1) Single-feature PDP ======
for feat in pdp_features:
    fig, ax = plt.subplots(figsize=(6, 4))
    PartialDependenceDisplay.from_estimator(
        xgb_full,
        X_vis,
        features=[feat],
        feature_names=X_vis.columns.tolist(),
        ax=ax,
        grid_resolution=200
    )
    ax.set_title(f"PDP - {feat}")
    plt.tight_layout()
    fig.savefig(os.path.join(output_dir, f"PDP_{feat}.png"))
    plt.close(fig)

# ====== 2) Pairwise interaction PDP ======
for f1, f2 in permutations(pdp_features, 2):
    fig, ax = plt.subplots(figsize=(6, 4))
    PartialDependenceDisplay.from_estimator(
        xgb_full,
        X_vis,
        features=[(f1, f2)],
        feature_names=X_vis.columns.tolist(),
        ax=ax,
        grid_resolution=50  # optional: coarser for speed
    )
    ax.set_title(f"PDP Interaction - {f1} & {f2}")
    plt.tight_layout()
    fname = f"PDP_{f1}_{f2}.png".replace(" ", "")
    fig.savefig(os.path.join(output_dir, fname))
    plt.close(fig)

print("PDP plots saved to:", output_dir)
