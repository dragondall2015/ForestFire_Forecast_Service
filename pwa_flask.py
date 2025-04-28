import os
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

# ─────────────────────── 1. Flask 기본 셋업 ───────────────────────
app = Flask(__name__)
app.config["SECRET_KEY"] = os.environ.get("SECRET_KEY", "dev-secret")
Bootstrap(app)
np.random.seed(42)

# ─────────────────────── 2. 모델 로드 (tf.keras) ─────────────────
model = model = tf.keras.models.load_model("fires_model_v3.keras", compile=False)

# ─────────────────────── 3. 전처리 파이프라인 ────────────────────
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

fires = pd.read_csv("sanbul2district-divby100.csv")
fires["burned_area"] = np.log(fires["burned_area"] + 1)
pipeline.fit(fires.drop(columns=["burned_area"]))

# ─────────────────────── 4. WTForms 입력 폼 ──────────────────────
class LabForm(FlaskForm):
    longitude       = StringField("longitude(1-7)"        , validators=[DataRequired()])
    latitude        = StringField("latitude(1-7)"         , validators=[DataRequired()])
    month           = StringField("month(01-Jan~12-Dec)"  , validators=[DataRequired()])
    day             = StringField("day(00-sun~07-hol)"    , validators=[DataRequired()])
    avg_temp        = StringField("avg_temp"              , validators=[DataRequired()])
    max_temp        = StringField("max_temp"              , validators=[DataRequired()])
    max_wind_speed  = StringField("max_wind_speed"        , validators=[DataRequired()])
    avg_wind        = StringField("avg_wind"              , validators=[DataRequired()])
    submit          = SubmitField("Submit")

# ─────────────────────── 5. 라우팅 ───────────────────────────────
@app.route("/", methods=["GET"])
def index():
    return render_template("index.html", form=LabForm())

@app.route("/prediction", methods=["GET", "POST"])
def prediction():
    form = LabForm()
    if form.validate_on_submit():
        df = pd.DataFrame([{
            "longitude"      : float(form.longitude.data),
            "latitude"       : float(form.latitude.data),
            "month"          : form.month.data,
            "day"            : form.day.data,
            "avg_temp"       : float(form.avg_temp.data),
            "max_temp"       : float(form.max_temp.data),
            "max_wind_speed" : float(form.max_wind_speed.data),
            "avg_wind"       : float(form.avg_wind.data)
        }])
        X = pipeline.transform(df)
        log_pred = model.predict(X, verbose=0)
        area     = np.exp(log_pred) - 1
        return render_template("result.html",
                               prediction=round(area[0][0], 2))
    return render_template("prediction.html", form=form)

# ─────────────────────── 6. 실행 엔트리 ──────────────────────────
if __name__ == "__main__":
    # Render는 $PORT 환경변수를 보내줍니다.
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=False)
