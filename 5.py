import keras
from keras.datasets import mnist
from matplotlib import pyplot as plt
import numpy as np

(x_train, y_train), (x_test, y_test) = mnist.load_data()

# Veamos la forma tiene x_train
print("Shape:", x_train.shape)  # 60.000 imágenes de 28x28

# Veamos una imagen cualquiera, por ejemplo, con el índice 125
image = np.array(x_train[125], dtype='float')
plt.imshow(image, cmap='gray')
plt.show()

print("Label:", y_train[125])

print("Max value:", max(x_train[125].reshape(784)))
print("Min value:", min(x_train[125].reshape(784)))


x_train = x_train.astype('float32')
x_test = x_test.astype('float32')

x_train /= 255  # Escalamos a un rango entre 0 y 1
x_test /= 255

x_train -= 0.5  # desplazamos el rango a -0.5 y 0.5
x_test -= 0.5

print("Max value:", max(x_train[125].reshape(784)))
print("Min value:", min(x_train[125].reshape(784)))

x_train = x_train.reshape(60000, 784)
x_test = x_test.reshape(10000, 784)