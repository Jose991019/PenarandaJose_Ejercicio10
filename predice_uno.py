import matplotlib.pyplot as plt
import sklearn.datasets as skdata
import numpy as np
import sklearn

numeros = skdata.load_digits()
target = numeros['target']
imagenes = numeros['images']
n_imagenes = len(target)

data = imagenes.reshape((n_imagenes, -1))

from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

scaler = StandardScaler()
x_train, x_test, y_train, y_test = train_test_split(data, target, train_size=0.7)

x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)

numero = 1
dd = y_train==numero
cov = np.cov(x_train[dd].T)
valores, vectores = np.linalg.eig(cov)
valores = np.real(valores)
vectores = np.real(vectores)
ii = np.argsort(-valores)
valores = valores[ii][:10]
vectores = vectores[:,ii][:,:10]


def predecir(imagen):
    es_uno = 0
    imagen = imagen[ii]
    proyeccion = np.dot(imagen,vectores)
    distancia = np.sqrt(np.sum(proyeccion-valores)**2)
    if(distancia<=60.0):
        es_uno = 1
    return(es_uno)



predict_xtrain = []
for i in x_train:
    a = predecir(i)
    predict_xtrain.append(a)
predict_xtest = []
for i in x_test:
    a = predecir(i)
    predict_xtest.append(a)
    
y_train[y_train > 1] = 0
y_test[y_test > 1] = 0

f1_train = sklearn.metrics.f1_score(y_train,predict_xtrain)
f1_test = sklearn.metrics.f1_score(y_test,predict_xtest)
plt.figure(figsize = (10,10))
plt.subplot(1,2,1)
plt.imshow(sklearn.metrics.confusion_matrix(y_train,predict_xtrain))
plt.title('F1Train = {}'.format(f1_train))
plt.subplot(1,2,2)
plt.imshow(sklearn.metrics.confusion_matrix(y_test,predict_xtest))
plt.title('F1Test = {}'.format(f1_test))
plt.savefig('matriz_de confusión.png')