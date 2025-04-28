# Keras용 라이브러리
import numpy as np
import tensorflow as tf
from tensorflow import keras
from sklearn.model_selection import train_test_split

# 데이터 불러오기
fires_prepared = np.load('./fires_prepared.npy')
fires_labels = np.load('./fires_labels.npy')
fires_test_prepared = np.load('./fires_test_prepared.npy')
fires_test_labels = np.load('./fires_test_labels.npy')

print("✅ 2020810041/손영준: 데이터 불러오기 완료")

# train/validation 분리
X_train, X_valid, y_train, y_valid = train_test_split(fires_prepared, fires_labels, test_size=0.2, random_state=42)
X_test, y_test = fires_test_prepared, fires_test_labels

# 랜덤 시드 고정
np.random.seed(42)
tf.random.set_seed(42)

# Keras 모델 생성
model = keras.models.Sequential([
    keras.layers.Dense(30, activation='relu', input_shape=X_train.shape[1:]),
    keras.layers.Dense(30, activation='relu'),
    keras.layers.Dense(30, activation='relu'),
    keras.layers.Dense(1)
])

model.summary()

# 모델 컴파일
model.compile(
    loss='mean_squared_error',
    optimizer=keras.optimizers.SGD(learning_rate=1e-3)
)

# 모델 학습
history = model.fit(
    X_train, y_train,
    epochs=200,
    validation_data=(X_valid, y_valid)
)

# 모델 저장
model.save('fires_model.keras')

# 모델 평가 예시
X_new = X_test[:3]
print("✅ 예측 결과 (소수 둘째자리까지 반올림):")
print(np.round(model.predict(X_new), 2))
