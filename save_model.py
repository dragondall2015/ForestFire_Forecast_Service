# save_model_v3.py
import tensorflow as tf

# 기존 모델 로드
model = tf.keras.models.load_model("fires_model.keras", compile=False)

# 새 포맷(Keras 3 compatible)으로 저장
model.save("fires_model_v3.keras")

print("✅ 모델을 Keras 3 포맷으로 저장했습니다: fires_model_v3.keras")
