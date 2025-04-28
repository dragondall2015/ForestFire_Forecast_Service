# pwa_flask.py
import os
import numpy as np
import pandas as pd
import tensorflow as tf
import joblib

from flask import Flask, render_template
from flask_bootstrap import Bootstrap
from flask_wtf import FlaskForm
from wtforms import StringField, SubmitField
from wtforms.validators import DataRequired

# ─────────────────────── 1. Flask 기본 셋업 ───────────────────────
app = Flask(__name__)
app.config["SECRET_KEY"] = os.environ.get("SECRET_KEY", "dev-secret")
Bootstrap(app)
np.random.seed(42)

# ─────────────────────── 2. 모델 로드 (tf.keras) ─────────────────
MODEL_PATH = "fires_model_v3.keras"
model = tf.keras.models.load_model(MODEL_PATH, compile=False)

# ─────────────────────── 3. 전처리 파이프라인 (load만)
pipeline = joblib.load("pipeline.pkl")

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
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=False)
