import tensorflow as tf
import numpy as np
import emnist
import cv2 as cv
import matplotlib.pyplot as plt
from extra_keras_datasets import emnist

(xTrain, yTrain), (xValidate, yValidate) = emnist.load_data(type='letters')
print(xTrain.shape)
print(yTrain.shape)
print(np.max(yTrain))
print(xValidate.shape)
print(yValidate.shape)


string = " abcdefghijklmnopqrstuvwxyz"
array = []
for c in string:
    array += [c]

# xTrain, yTrain = xTrain / 255.0, yTrain / 255.0

# normalize the x values of test and train from dataset - no need to do the y values
xTrain = tf.keras.utils.normalize(xTrain, axis=1)
# yTrain = tf.keras.utils.normalize(yTrain, axis=1)

model = tf.keras.models.Sequential()
model.add(tf.keras.layers.Flatten(input_shape = (28,28,1)))
model.add(tf.keras.layers.Dense(units=256, activation=tf.nn.relu))
model.add(tf.keras.layers.Dense(units=128, activation=tf.nn.relu))
model.add(tf.keras.layers.Dense(units=64, activation=tf.nn.relu))
model.add(tf.keras.layers.Dense(units=27, activation=tf.nn.softmax))

model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

model.fit(xTrain,yTrain,epochs=5)

loss, accuracy = model.evaluate(xValidate, yValidate)
print("acurracy is " , accuracy)
print("loss is ", loss)

model.save('digits.model')


img = cv.imread('lettern.png')[:, :, 0]

img = np.expand_dims(img, axis=0)

print(img.shape)
img = tf.keras.utils.normalize(img, axis=1)
prediction = model.predict(img)
print(prediction.shape)
prediction = np.squeeze(prediction)
print(f'The result is probably: {array[np.argmax(prediction)]}')
print(f'The probability is :  {prediction[np.argmax(prediction)]}')
plt.imshow(img[0], cmap=plt.cm.binary)
plt.show()

