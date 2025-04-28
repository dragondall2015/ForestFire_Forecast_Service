import numpy as np
import pandas as pd
import tensorflow as tf
from flask import Flask, render_template
from flask_bootstrap import Bootstrap
from flask_wtf import FlaskForm
from wtforms import StringField, SubmitField
from wtforms.validators import DataRequired
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
import os


# ─────────────────────────────────────────────────────────
# 1. 앱 & 기본 설정
# ─────────────────────────────────────────────────────────
app = Flask(__name__)
app.config["SECRET_KEY"] = os.environ.get("SECRET_KEY", "dev-secret")  # fallback
bootstrap = Bootstrap(app)
np.random.seed(42)
# ─────────────────────────────────────────────────────────
# 2. 학습한 Keras 모델 로드
# ─────────────────────────────────────────────────────────
model = tf.keras.models.load_model("fires_model.keras", compile=False)
# ─────────────────────────────────────────────────────────
# 3. 전처리 파이프라인 준비  (handle_unknown="ignore" 추가!)
# ─────────────────────────────────────────────────────────
num_attribs = [
    "longitude", "latitude", "avg_temp",
    "max_temp", "max_wind_speed", "avg_wind"
]
cat_attribs = ["month", "day"]

def build_pipeline():
    num_pipe = Pipeline([("scaler", StandardScaler())])
    full     = ColumnTransformer([
        ("num", num_pipe, num_attribs),
        ("cat", OneHotEncoder(handle_unknown="ignore"), cat_attribs)
    ])
    return full

pipeline = build_pipeline()

# fit 1회
fires = pd.read_csv("./sanbul2district-divby100.csv")
fires["burned_area"] = np.log(fires["burned_area"] + 1)
pipeline.fit(fires.drop(["burned_area"], axis=1))
# ─────────────────────────────────────────────────────────
# 4. 입력 폼 정의
# ─────────────────────────────────────────────────────────
class LabForm(FlaskForm):
    longitude       = StringField("longitude(1-7)"      , validators=[DataRequired()])
    latitude        = StringField("latitude(1-7)"       , validators=[DataRequired()])
    month           = StringField("month(01-Jan~12-Dec)", validators=[DataRequired()])
    day             = StringField("day(00-sun~07-hol)" , validators=[DataRequired()])
    avg_temp        = StringField("avg_temp"            , validators=[DataRequired()])
    max_temp        = StringField("max_temp"            , validators=[DataRequired()])
    max_wind_speed  = StringField("max_wind_speed"      , validators=[DataRequired()])
    avg_wind        = StringField("avg_wind"            , validators=[DataRequired()])
    submit          = SubmitField("Submit")
# ─────────────────────────────────────────────────────────
# 5. 라우트
# ─────────────────────────────────────────────────────────
@app.route("/", methods=["GET"])
@app.route("/index", methods=["GET"])
def index():
    return render_template("index.html", form=LabForm())

@app.route("/prediction", methods=["GET", "POST"])
def prediction():
    form = LabForm()
    if form.validate_on_submit():
        # 입력값 → DataFrame
        input_df = pd.DataFrame([{
            "longitude"      : float(form.longitude.data),
            "latitude"       : float(form.latitude.data),
            "month"          : form.month.data,
            "day"            : form.day.data,
            "avg_temp"       : float(form.avg_temp.data),
            "max_temp"       : float(form.max_temp.data),
            "max_wind_speed" : float(form.max_wind_speed.data),
            "avg_wind"       : float(form.avg_wind.data)
        }])
        # 전처리(transform만)
        X = pipeline.transform(input_df)
        # 예측 및 역-로그 변환
        log_pred = model.predict(X)
        area_pred = np.exp(log_pred) - 1
        return render_template(
            "result.html",
            prediction=round(area_pred[0][0], 2)
        )
    return render_template("prediction.html", form=form)
# ─────────────────────────────────────────────────────────
# 6. 서버 실행
# ─────────────────────────────────────────────────────────
if __name__ == "__main__":
    app.run(debug=True)