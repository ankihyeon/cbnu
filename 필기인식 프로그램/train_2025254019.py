import tensorflow as tf
from tensorflow.keras.datasets import mnist
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten, Conv2D, MaxPooling2D
from tensorflow.keras.optimizers import Adam

# MNIST 데이터셋 로드
(train_images, train_labels), (test_images, test_labels) = mnist.load_data()

# 데이터 전처리
train_images = (
    train_images.reshape((train_images.shape[0], 28, 28, 1)).astype("float32") / 255
)
test_images = (
    test_images.reshape((test_images.shape[0], 28, 28, 1)).astype("float32") / 255
)

# 모델 구성
model = Sequential(
    [
        Conv2D(32, (3, 3), activation="relu", input_shape=(28, 28, 1)),
        MaxPooling2D((2, 2)),
        Flatten(),
        Dense(128, activation="relu"),
        Dense(10, activation="softmax"),
    ]
)

# 모델 컴파일
model.compile(
    optimizer=Adam(), loss="sparse_categorical_crossentropy", metrics=["accuracy"]
)

# 모델 학습
model.fit(
    train_images,
    train_labels,
    epochs=5,
    batch_size=64,
    validation_data=(test_images, test_labels),
)

# 모델 저장
model.save("2025254019.h5")
