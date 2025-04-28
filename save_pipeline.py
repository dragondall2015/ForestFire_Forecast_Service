# save_pipeline.py
import numpy as np
import pandas as pd
import joblib
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer

# ────────────────────────────────────────────────────
# 1. 전처리 파이프라인 구성
# ────────────────────────────────────────────────────
num_cols = ["longitude", "latitude", "avg_temp",
            "max_temp", "max_wind_speed", "avg_wind"]
cat_cols = ["month", "day"]

def build_pipeline():
    num_pipe = Pipeline([("scaler", StandardScaler())])
    return ColumnTransformer([
        ("num", num_pipe, num_cols),
        ("cat", OneHotEncoder(handle_unknown="ignore"), cat_cols)
    ])

pipeline = build_pipeline()

# ────────────────────────────────────────────────────
# 2. 데이터 읽고 fit
# ────────────────────────────────────────────────────
fires = pd.read_csv("sanbul2district-divby100.csv")
fires["burned_area"] = np.log(fires["burned_area"] + 1)

pipeline.fit(fires.drop(columns=["burned_area"]))

# ────────────────────────────────────────────────────
# 3. 저장
# ────────────────────────────────────────────────────
joblib.dump(pipeline, "pipeline.pkl")

print("✅ pipeline.pkl 저장 완료!")
