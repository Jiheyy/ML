import tensorflow as tf
import numpy as np
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense

def main():
    dense = Dense(units=1, input_shape=[1])
    model = Sequential([dense])
    model.compile(optimizer="sgd", loss="mean_squared_error")

    xs = np.array([-1.0, 0.0, 1.0, 2.0, 3.0, 4.0], dtype=float)
    xy = np.array([-3.0, -1.0, 1.0, 3.0, 5.0, 7.0], dtype=float)

    model.fit(xs, xy, epochs=500) # 훈련과정 시작 . X와 y를 통해 훈련하고 500 번 반복하라
    # 첫번째 반복에서 컴퓨터는 관계를 추측하고 이 추측이 얼마나 좋고 나쁜지 측정한다.
    # 그 다음 이 결과를 옵티마이저에 피드백해 새로운 추측을 생성한다.
    # 손실(또는 오차) 가 시간이 지남에 따라 줄어드는 로직을 사용해 이 과정을 반복한다.
    # 결과적으로 이 추측은 점점 더 좋아진다.

    print(model.predict([10.0]))
    print("신경망이 학습한 것  {}".format(dense.get_weights()))


if __name__ == '__main__':
    main()
