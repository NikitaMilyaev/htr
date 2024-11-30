import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
'''
from tensorflow.keras.datasets import mnist
from tensorflow.keras.layers import Dense, Flatten

(x_train, y_train), (x_test, y_test) = mnist.load_data()

x_train = tf.keras.utils.normalize(x_train, axis=1)
x_test = tf.keras.utils.normalize(x_test, axis=1)


model = tf.keras.models.Sequential()
model.add(tf.keras.layers.Flatten(input_shape=(28, 28)))
model.add(tf.keras.layers.Dense(128, activation='relu'))
model.add(tf.keras.layers.Dense(128, activation='relu'))
model.add(tf.keras.layers.Dense(10, activation='softmax'))

model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])


model.fit(x_train, y_train, batch_size=32, epochs=7, validation_split=0.1)
model.save('handwrittenRecognition.keras')
'''
model = tf.keras.models.load_model('handwrittenRecognition.keras')
'''
loss, accuracy = model.evaluate(x_test, y_test)

print("Loss : ", loss)
print("Accuracy : ", accuracy)
'''
image_number = 1

#n = 0

while os.path.isfile(f"digit{image_number}.png"):
    try:
        img = cv2.imread(f"digit{image_number}.png")[:,:,0]
        img = cv2.resize(img, (28,28))
        img = np.invert(np.array([img]))
        img = img/255
        prediction = model.predict(img)
        print("The number is probably a {}".format(np.argmax(prediction)))
        plt.imshow(img[0], cmap=plt.cm.binary)
        plt.show()
    except:
        print(f"error: {image_number}")
    finally:
        image_number += 1