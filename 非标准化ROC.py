import os
import joblib
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc
from sklearn.model_selection import train_test_split

# ====== Paths ======
base_dir  = "/mnt/29T_HardDisk/huzihan_private/广东省第二人民医院/杨主任论文"
model_dir = os.path.join(base_dir, "非标准化模型")
data_X    = os.path.join(base_dir, "X_con.csv")
data_y    = os.path.join(base_dir, "y_con.csv")
viz_dir   = os.path.join(base_dir, "非标准化可视化图片")
os.makedirs(viz_dir, exist_ok=True)

# ====== 1. Load data and split ======
X = pd.read_csv(data_X)
y = pd.read_csv(data_y).squeeze("columns")
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42
)

# ====== 2. Load trained models ======
models = {}
for fname in os.listdir(model_dir):
    if fname.endswith("_model.pkl"):
        name = fname.replace("_model.pkl", "")
        models[name] = joblib.load(os.path.join(model_dir, fname))

# ====== 3. Plot ROC for each pre-trained model ======
plt.figure(figsize=(8, 6))
for name, model in models.items():
    # get predicted probability for positive class
    if hasattr(model, "predict_proba"):
        y_score = model.predict_proba(X_test)[:, 1]
    else:
        # fallback to decision_function, then normalize
        dfun    = model.decision_function(X_test)
        y_score = (dfun - dfun.min()) / (dfun.max() - dfun.min())

    fpr, tpr, _ = roc_curve(y_test, y_score)
    roc_auc = auc(fpr, tpr)
    plt.plot(fpr, tpr, lw=2, label=f"{name} (AUC={roc_auc:.2f})")

# diagonal chance line
plt.plot([0, 1], [0, 1], linestyle="--", color="gray", lw=1, label="Chance")

plt.xlim(0, 1)
plt.ylim(0, 1.05)
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curve Comparison (Pre-trained Models)")
plt.legend(loc="lower right")
plt.tight_layout()

# save
out_path = os.path.join(viz_dir, "ROC_Pretrained_Comparison.png")
plt.savefig(out_path)
plt.close()

print("Loaded models:", list(models.keys()))
print("ROC comparison saved to:", out_path)
