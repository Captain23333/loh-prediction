# app.py
from flask import Flask, render_template_string, request
import os, joblib
import pandas as pd
from sklearn.preprocessing import OrdinalEncoder

app = Flask(__name__)

# ====== 1) 特征映射 ======
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
    '文化程度1：低于 9 年级，2：9-11 年级（包括 12 年级，无文凭），3：高中毕业/GED 或同等学历，4：一些大学或 AA 学位，5：大专以上学历': 'education_level',
    '婚姻1：已婚，2：丧偶，3：离婚，4：分离，5：从未结婚，6：与伴侣同住': 'marital_status',
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

binary_cols = [
    '喝酒（1-喝酒）', '冠心病（1-有）',
    '心绞痛（1-有）', '吸烟（1-有）', '性别（1男）'
]

ordinal_cols = [
    '糖尿病（1-有，2-无，3临界糖尿病）',
    '种族（1-墨西哥裔美国人，2-其他西班牙人，3-白人，4-黑人，5-其他）',
    '文化程度1：低于 9 年级，2：9-11 年级（包括 12 年级，无文凭），3：高中毕业/GED 或同等学历，4：一些大学或 AA 学位，5：大专以上学历',
    '婚姻1：已婚，2：丧偶，3：离婚，4：分离，5：从未结婚，6：与伴侣同住'
]

# ====== 2) 有序字段英文选项 ======
ordinal_options = {
    '糖尿病（1-有，2-无，3临界糖尿病）': ['Diabetes', 'No Diabetes', 'Borderline Diabetes'],
    '种族（1-墨西哥裔美国人，2-其他西班牙人，3-白人，4-黑人，5-其他）': [
        'Mexican American', 'Other Hispanic', 'White', 'Black', 'Other'
    ],
    '文化程度1：低于 9 年级，2：9-11 年级（包括 12 年级，无文凭），3：高中毕业/GED 或同等学历，4：一些大学或 AA 学位，5：大专以上学历': [
        'Less than 9th grade',
        '9–11th grade (incl. 12th, no diploma)',
        'High school grad/GED',
        'Some college/AA degree',
        'College graduate or above'
    ],
    '婚姻1：已婚，2：丧偶，3：离婚，4：分离，5：从未结婚，6：与伴侣同住': [
        'Married',
        'Widowed',
        'Divorced',
        'Separated',
        'Never married',
        'Living with partner'
    ]
}

# 与上面顺序一致，用于编码
ord_categories = list(ordinal_options.values())
oe = OrdinalEncoder(categories=ord_categories)

# ====== 3) 加载模型 ======
base_dir   = "loh-prediction"
model_path = os.path.join(base_dir, "非标准化模型", "XGBoost_model_nonstd.pkl")
xgb        = joblib.load(model_path)

# ====== 4) HTML 模板 (Bootstrap) ======
FORM = """
<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <title>LOH Risk Prediction</title>
  <link href="https://cdn.jsdelivr.net/npm/bootstrap@5.3.0/dist/css/bootstrap.min.css"
        rel="stylesheet">
</head>
<body class="p-4">
  <div class="container">
    <h1 class="mb-4">LOH Risk Prediction</h1>
    <form method="post">
      {% for chi, eng in rename_dict.items() %}
        <div class="mb-3 row">
          <label class="col-sm-4 col-form-label">{{ eng }}</label>
          <div class="col-sm-8">
            {% if chi in binary_cols %}
              <select class="form-select" name="{{ chi }}">
                <option value="1">Yes</option>
                <option value="0">No</option>
              </select>
            {% elif chi in ordinal_cols %}
              <select class="form-select" name="{{ chi }}">
                {% for opt in ordinal_options[chi] %}
                  <option value="{{ opt }}">{{ opt }}</option>
                {% endfor %}
              </select>
            {% else %}
              <input type="number" step="any" class="form-control" name="{{ chi }}" required>
            {% endif %}
          </div>
        </div>
      {% endfor %}
      <button type="submit" class="btn btn-primary">Predict</button>
    </form>

    {% if prob is defined %}
      <div class="alert alert-success mt-4">
        Predicted LOH risk probability: <strong>{{ prob|round(4) }}</strong>
      </div>
    {% endif %}
  </div>
</body>
</html>
"""

@app.route("/", methods=["GET","POST"])
def predict():
    context = {
        'rename_dict':     rename_dict,
        'binary_cols':     binary_cols,
        'ordinal_cols':    ordinal_cols,
        'ordinal_options': ordinal_options
    }

    if request.method == "POST":
        # 1) 收集原始（中文）表单字段
        raw = { chi: request.form[chi] for chi in rename_dict }

        # 2) 构 DataFrame，重命名到英文
        df_raw = pd.DataFrame([raw])
        df     = df_raw.rename(columns=rename_dict)

        # 3) 转型
        for chi in binary_cols:
            df[rename_dict[chi]] = df[rename_dict[chi]].astype(int)
        for chi in ordinal_cols:
            df[rename_dict[chi]] = df_raw[chi]  # 字符串
        for chi, eng in rename_dict.items():
            if chi not in binary_cols + ordinal_cols:
                df[eng] = df[eng].astype(float)

        # 4) 有序编码
        ord_eng = [rename_dict[c] for c in ordinal_cols]
        df[ord_eng] = oe.fit_transform(df[ord_eng])

        # 5) 预测概率
        prob = xgb.predict_proba(df[list(rename_dict.values())])[:,1][0]
        context['prob'] = prob

    return render_template_string(FORM, **context)

if __name__=="__main__":
    app.run(host="0.0.0.0", port=5000)
