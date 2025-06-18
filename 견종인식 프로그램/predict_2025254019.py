import cv2 as cv
import numpy as np
import tensorflow as tf
import pickle

# 1) 모델 및 클래스 로드
cnn = tf.keras.models.load_model('./2025254019.h5')
with open('./dog_species_name.txt', 'rb') as f:
    dog_species = pickle.load(f)

# 2) 평가용 이미지 불러오기
img_path = './data/Images/n02094433-Yorkshire_terrier/n02094433_800.jpg'
img = cv.imread(img_path)
resized = cv.resize(img, (224, 224))
x = np.reshape(resized, (1, 224, 224, 3))

# 3) 예측 수행
res = cnn.predict(x)[0]
top5 = np.argsort(-res)[:5]
top5_dog_species = [dog_species[i] for i in top5]

# 4) 결과 이미지에 확률 및 견종 표시
for i in range(5):
    prob = '(' + str(round(res[top5[i]], 4)) + ')'
    name = str(top5_dog_species[i].split('-')[1])
    cv.putText(img, prob + name, (10, 100 + i * 30),
               cv.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

# 5) 최종 결과 저장
cv.imwrite('2025254019.png', img)
