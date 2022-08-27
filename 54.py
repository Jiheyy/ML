import tensorflow as tf
data = tf.keras.datasets.fashion_mnist

(training_images, training_labels), (test_images, test_labels) = data.load_data()

training_images = training_images / 255.0
test_images = test_images / 255.0

model = tf.keras.models.Sequential([
    tf.keras.layers.Flatten(input_shape=(28, 28)),
    tf.keras.layers.Dense(128, activation=tf.nn.relu),
    tf.keras.layers.Dense(10, activation=tf.nn.softmax)
])

model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

model.fit(training_images, training_labels, epochs=50)
# 과대적합
# 훈련 세트의 정확도는 크게 높아졌지만 테스트 세트에 대한 정확도는 조금만 향상이 되었음
# 오래 훈련했기 때무에 더 높은 성능이 나오기는 했지만 월등하게 더 나은 모델은 아니다.
# 사실 정확도 차이가 더 커졌으므로 모델이 훈련 세트에 특화되었음을 보여준다.
# 이를 과대적합이라고 한다.

#model.evaluate(test_images, test_labels)