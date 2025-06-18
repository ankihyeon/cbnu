import tensorflow as tf
import numpy as np
import cv2

# 학습된 모델 로드
model = tf.keras.models.load_model("2025254019.h5")

# 평가용 이미지 로드 (여기서는 데이터셋에서 평가용 이미지를 선택)
img_path = "./testData.png"  # 예시 이미지 경로
img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
img_resized = cv2.resize(img, (28, 28))
img_normalized = img_resized.reshape(1, 28, 28, 1).astype("float32") / 255

# 숫자 예측
predicted_class = np.argmax(model.predict(img_normalized), axis=-1)

# 결과 출력
with open("2025254019.txt", "w") as f:
    f.write(str(predicted_class[0]))