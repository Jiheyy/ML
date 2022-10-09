import sys

if 'google.colab' in sys.modules:
    from google.colob import files
    uploaded = files.uplodad()
    sample_images = ['/content/'+ fn for fn in uploaded.keys()]

    if len(uploaded) < 1:
        import gdown
        base_url = 'https://github.com/rickiepark/aiml4coders/tree/main/ch03'
        for i in range(1, 4):
            gdown.download(base_url + 'hh_image_{}.jpg'.format(i))
        sample_images = ['/content/hh_image_{}.jpg'.format(i) for i in range(1, 4)]
    else:
        sample_images = ['hh_image_{}.jpg'.format(i) for i in range(1, 4)]

### 84
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from keras.preprocessing import image

for fn in sample_images:
    plt.imshow(mpimg.imread(fn))
    plt.show()

    # 이미지 불러오기
    img = tf.keras.utils.load_img(fn, target_size=(300, 300))
    x = tf.keras.utils.img_to_array(img)
    x = np.expand_dims(x, axis=0)

    classes = model.predict(x)

    print(classes[0][0])
    if(classes[0][0] > 0.5):
        print('human')
    else:
        print('horse')

    print('-----------')