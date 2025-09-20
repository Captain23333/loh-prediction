import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import OrdinalEncoder
from imblearn.over_sampling import RandomOverSampler
from tqdm import tqdm
import warnings

warnings.filterwarnings("ignore")

# ====== 自定义列名 ======
continuous_cols = [
    '脉率', '收缩压', '舒张压', '体重（千克）', '身高（厘米）',
    '体质指数（千克/米²）', '手臂围（厘米）', '腰围（厘米）',
    'HDL（mmol/L）', '甘油三酯 (mg/dL)', '甘油三酯（mmol/L）',
    'LDL（mmol/L）', '总胆固醇（mmol/L）',
    '白细胞计数（1000细胞/μl）', '血红蛋白 (g/dL)', '血细胞比容 (%)',
    '平均红细胞体积 (fL)', '血小板计数（1000细胞/μl）',
    '糖化血红蛋白（%）', '空腹血糖（mg/dL）', '空腹血糖（mmol/L）',
    'SHBG', '尿酸（mg/dL）', '维生素D3（nmol/L）',
    '年龄', '贫困比', 'TyG 指数', 'TyG-BMI指数',
    '腰高比', 'TyG-腰高比指数', 'AIP',
    '尿酸与高密度脂蛋白胆固醇比值', '心脏代谢指数',
    '身体圆度指数', '体重调整后的腰围指数'
]

binary_cols = [
    '喝酒（1-喝酒）', '冠心病（1-有）', '心绞痛（1-有）',
    '吸烟（1-有）', '性别（1男）'
]

ordinal_cols = [
    '糖尿病（1-有，2-无，3临界糖尿病）',
    '种族（1-墨西哥裔美国人，2-其他西班牙人，3-白人，4-黑人，5-其他）',
    '文化程度1：低于 9 年级，2：9-11 年级（包括 12 年级，无文凭），3：高中毕业/GED 或同等学历，4：一些大学或 AA 学位，5：大专以上学历',
    '婚姻1：已婚，2：丧偶，3：离婚，4：分离，5：从未结婚，6：与伴侣同住'
]

# ====== 重命名字典 ======
rename_dict = {
    '脉率': 'pulse_rate',
    '收缩压': 'systolic_bp',
    '舒张压': 'diastolic_bp',
    '体重（千克）': 'weight_kg',
    '身高（厘米）': 'height_cm',
    '体质指数（千克/米²）': 'BMI',
    '手臂围（厘米）': 'arm_circumference_cm',
    '腰围（厘米）': 'waist_circumference_cm',
    'HDL（mmol/L）': 'HDL_mmol_per_L',
    '甘油三酯 (mg/dL)': 'triglycerides_mg_per_dL',
    '甘油三酯（mmol/L）': 'triglycerides_mmol_per_L',
    'LDL（mmol/L）': 'LDL_mmol_per_L',
    '总胆固醇（mmol/L）': 'total_cholesterol_mmol_per_L',
    '白细胞计数（1000细胞/μl）': 'WBC_count',
    '血红蛋白 (g/dL)': 'hemoglobin_g_per_dL',
    '血细胞比容 (%)': 'hematocrit_percent',
    '平均红细胞体积 (fL)': 'MCV_fL',
    '血小板计数（1000细胞/μl）': 'platelet_count',
    '糖化血红蛋白（%）': 'HbA1c_percent',
    '空腹血糖（mg/dL）': 'fasting_glucose_mg_per_dL',
    '空腹血糖（mmol/L）': 'fasting_glucose_mmol_per_L',
    'SHBG': 'SHBG',
    '尿酸（mg/dL）': 'uric_acid_mg_per_dL',
    '维生素D3（nmol/L）': 'vitamin_D3_nmol_per_L',
    '喝酒（1-喝酒）': 'drinking',
    '糖尿病（1-有，2-无，3临界糖尿病）': 'diabetes_status',
    '冠心病（1-有）': 'coronary_heart_disease',
    '心绞痛（1-有）': 'angina',
    '吸烟（1-有）': 'smoking',
    '性别（1男）': 'male',
    '年龄': 'age',
    '种族（1-墨西哥裔美国人，2-其他西班牙人，3-白人，4-黑人，5-其他）': 'race_ethnicity',
    '文化程度1：低于 9 年级，2： 9-11 年级（包括 12 年级，无文凭），3：高中毕业/GED 或同等学历，4：一些大学或 AA 学位，5：大专以上学历': 'education_level',
    '婚姻1：已婚，2：丧偶，3： 离婚，4：分离，5：从未结婚，6： 与伴侣同住': 'marital_status',
    '贫困比': 'poverty_income_ratio',
    'TyG 指数': 'TyG_index',
    'TyG-BMI指数': 'TyG_BMI_index',
    '腰高比': 'waist_height_ratio',
    'TyG-腰高比指数': 'TyG_waist_height_index',
    'AIP': 'AIP',
    '尿酸与高密度脂蛋白胆固醇比值': 'uric_acid_to_HDL_ratio',
    '心脏代谢指数': 'cardiometabolic_index',
    '身体圆度指数': 'body_roundness_index',
    '体重调整后的腰围指数': 'weight_adjusted_waist_index'
}

# ====== Step 1: 读取数据 ======
file_path = r"/mnt/29T_HardDisk/huzihan_private/广东省第二人民医院/杨主任论文/nhanes-LOH2(1).xlsx"
df = pd.read_excel(file_path)

target_col = "总睾酮（ng/dL）（结局1-有LOH,2-无LOH）"
y = df[target_col].map(lambda x: 1 if x == 1 else 0)
X = df.drop(columns=[target_col])

# ====== Step 2: 手写 MICE + RF 插补 ======
def mice_rf_impute(X, max_iter=10, n_estimators=100):
    X_imp = X.copy()
    # 初始均值填充
    for c in X_imp.columns:
        X_imp[c].fillna(X_imp[c].mean(), inplace=True)
    features = X.columns.tolist()
    for it in tqdm(range(max_iter), desc="MICE Iter"):
        for col in tqdm(features, desc=f" Iter {it+1}", leave=False):
            mask = X[col].isna()
            if not mask.any():
                continue
            rf = RandomForestRegressor(n_estimators=n_estimators, random_state=it)
            rf.fit(X_imp.loc[~mask, features], X_imp.loc[~mask, col])
            X_imp.loc[mask, col] = rf.predict(X_imp.loc[mask, features])
    return X_imp

X_imp = mice_rf_impute(X, max_iter=10, n_estimators=100)

# ====== Step 3: 编码（不做标准化） ======

# 1. 二元 0/1 映射
for col in binary_cols:
    if col in X_imp:
        X_imp[col] = X_imp[col].map(lambda v: 1 if v == 1 else 0)

# 2. OrdinalEncoder（有序类别）
ord_present = [c for c in ordinal_cols if c in X_imp.columns]
if ord_present:
    oe = OrdinalEncoder()
    X_imp[ord_present] = oe.fit_transform(X_imp[ord_present])

# ====== Step 4: 重命名列 & 删除 SEQN ======
X_enc = X_imp.copy()
X_enc.rename(columns=rename_dict, inplace=True)
if 'SEQN' in X_enc.columns:
    X_enc.drop(columns=['SEQN'], inplace=True)

# ====== Step 5: 保存 ======
X_enc.to_csv(
    "/mnt/29T_HardDisk/huzihan_private/广东省第二人民医院/杨主任论文/X_con.csv",
    index=False
)
y.to_csv(
    "/mnt/29T_HardDisk/huzihan_private/广东省第二人民医院/杨主任论文/y_con.csv",
    index=False,
    header=True
)
