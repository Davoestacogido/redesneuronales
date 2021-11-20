import numpy as np
import nn4t


data = np.genfromtxt('iris.data', delimiter=",")  # Cargamos el dataset iris
np.random.shuffle(data)  # Desordenamos el dataset para conseguir minibatchs con suficiente variación de muestras

x_data = data[:, 0:4].astype('f4')  # Muestras

y_data = nn4t.one_hot(data[:, 4].astype(int), 3)  # Etiquetas en formato one-hot

batch_size = 15  # tamaño del minibatch
epochs = 100  # Número de épocas

net = nn4t.Net(layers=[4, 5, 3])  # Configurasmos nuestra red con 4 entradas, 5 neuronas en la capa oculta y 3 salidas
error=0
for _ in range(epochs):  # Bucle de épocas
    for i in range(int(data.shape[0]/batch_size)):  # Número de minilotes en el dataset
        p = i*batch_size  # Índice del próximo minilote
        net.train(x_data[p:p+batch_size], y_data[p:p+batch_size])  # Entrenamos la red. Esto significa actualizar los pesos con los gradientes calculados
    for j in range(150):
        for x, y in zip(x_data[:150], y_data[:150]):
            error+= (net.output(x)-y)**2
print(error/150)


aciertos=0
for x, y in zip(x_data[:150], y_data[:150]): # Para esta prueba utilizaremos solamente las 15 primeras muestras
    print(y, "-->", net.output(x))  # Comparamos con las etiquetas
    eluno = np.searchsorted(y, 1)
    if net.output(x)[0]>net.output(x)[1]:
        if net.output(x)[0]>net.output(x)[2]:
            if eluno==3:
                aciertos+=1
        else:
            if eluno==2:
                aciertos+=1
    elif net.output(x)[1]>net.output(x)[2]:
        if eluno == 1:
            aciertos += 1
    else:
        if eluno ==2:
            aciertos+=1
print(aciertos)



