import numpy as np
from matplotlib import pyplot as plt

features = [[10, 16, 8],  # megapixeles de la cámara
            [64, 256, 32]]  # gigabytes de memoria

classes = []
for e1, e2 in zip(features[0], features[1]):
    if e1 > 15 and e2 > 128:
        classes.append('b')
    else:
        classes.append('r')

plt.scatter(features[0], features[1], c=classes)

plt.xlabel("Eje X: Megapixeles de la cámara")
plt.ylabel("Eje Y: Gigabytes de memoria");

plt.show()

#desciendo por el gradiente
def f(x):
    return x ** 2 - 2 * x + 2

x = 2.501  # algún punto inicial, donde empieza a buscar el minimo
delta = 0.01  # algún valor pequeño, los pasitos

counter = 0
while (f(x) - f(x - delta)) > 0: # avanza un pasito hacia el centro y comprueba si llego al minimo
    x -= delta  # nuevo x
    counter += 1

print("Aproximación al mínimo:", x)
print("Pasos:", counter)

features = [[1, 2],
            [3, 1],
            [4, 5]]

labels = [1, 1, 0]


def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))


def sigmoid_derivate(o):
    return o * (1.0 - o)


def train(x_data, y_data):
    np.random.seed(seed=123)
    w0, w1, w2 = np.random.rand(3)
    lr = 0.1 #lenin reight
    epochs = 10000

    print("Training...")

    for _ in range(epochs):

        w0_d = []
        w1_d = []
        w2_d = []

        for data, label in zip(x_data, y_data):
            e1, e2 = data;
            o = sigmoid(w0 * 1.0 + w1 * e1 + w2 * e2)
            aux = 2. * (o - label) * sigmoid_derivate(o)

            w0_d.append(aux * 1.0)
            w1_d.append(aux * e1)
            w2_d.append(aux * e2)

        w0 = w0 - np.sum(w0_d) * lr
        w1 = w1 - np.sum(w1_d) * lr
        w2 = w2 - np.sum(w2_d) * lr

    for data, label in zip(x_data, y_data):
        e1, e2 = data;
        print(data, "->", label)
        o = sigmoid(w0 * 1.0 + w1 * e1 + w2 * e2)
        print(o)
        print("-----------------------")

    print("Pesos: ", w0, w1, w2)


train(features, labels)