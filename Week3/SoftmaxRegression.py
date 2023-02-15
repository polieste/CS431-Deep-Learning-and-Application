import numpy as np
import matplotlib.pyplot as plt
from keras.layers import Dense, Input
from keras import Model,models
import tensorflow as tf
import random

#b1 generate data
means = [[1, 2], [2, 3], [3, 6]]
cov = [[1, 0], [0, 1]]
N = 100
X0 = np.random.multivariate_normal(means[0], cov, N)
X1 = np.random.multivariate_normal(means[1], cov, N)
X2 = np.random.multivariate_normal(means[2], cov, N)
X = np.concatenate((X0, X1,X2), axis = 0)
print(X.shape)
Y= np.asarray([0]*N + [1]*N+[2]*N).T
plt.scatter(X0[:,0],X0[:,1],c='#d62728')
plt.scatter(X1[:,0],X1[:,1],c='#9467bd')
plt.scatter(X2[:,0],X2[:,1],c='#17becf')
plt.show()

#b2 code model vs 8 phuong thuc
class softmax_regression:
    def __init__(self):
        return None
    def build(self,in_dim=2):
        input=Input(in_dim)
        output=Dense(3,activation='softmax',use_bias=True)(input)
        self.model=Model(input,output)
        return self.model
    def train(self,x_train,y_train):
        self.model.compile(optimizer='SGD',loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True))
        hist=self.model.fit(x_train,y_train,epochs=1000)
        return hist
    def save(self,path_model):
        return self.model.save(path_model)
    def load(self,path_model):
        return models.load_model(path_model)
    def predict(self,x_test):
        return self.model.predict(x_test)
    def summary(self):
        return self.model.summary()
    def get_paramater(self):
        return self.model.get_weights()
#b3 khoi tao va train
softmax_Model=softmax_regression()
softmax_Model.build()
hist=softmax_Model.train(X,Y)
weights=softmax_Model.get_paramater()
print(weights)

#b4 visualize loss+ hyper plane
plt.plot(hist.history['loss'])
plt.show()



#Visualize
#x1_min, x1_max = X[:, 0].min() - .5, X[:, 0].max() + .5
#x2_min, x2_max = X[:, 1].min() - .5, X[:, 1].max() + .5
#print(x1_min,x1_max,x2_min,x2_max)#-0.86 11.025 -1.68 9.26


xm = np.arange(-2, 11, 0.025)
xlen = len(xm)
ym = np.arange(-3, 10, 0.025)
ylen = len(ym)
xx, yy = np.meshgrid(xm, ym)


# xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))
# xx.ravel(), yy.ravel()

xx1 = xx.ravel().reshape(1, xx.size)
yy1 = yy.ravel().reshape(1, yy.size)

print(xx.shape, yy.shape)
print(xx1.shape, yy1.shape)
XX = np.concatenate(( xx1, yy1), axis = 0)
print(XX.shape)
XX=XX.T
print(XX.shape)
Z=softmax_Model.predict(XX)
print(Z.shape)
Z1=[np.argmax(i) for i in Z]
Z1=np.array(Z1)
print(Z1[0:5])
Z1 = Z1.reshape(xx.shape)
print(Z1[0:5,0:5])
# plt.figure(1
# plt.pcolormesh(xx, yy, Z, cmap='jet', alpha = .35)

CS = plt.contourf(yy, xx, Z1, 200, cmap='jet', alpha = .1)

# Plot also the training points
# plt.scatter(X[:, 1], X[:, 2], c=Y, edgecolors='k', cmap=plt.cm.Paired)
# plt.xlabel('Sepal length')
# plt.ylabel('Sepal width')

plt.xlim(-2, 11)
plt.ylim(-3, 10)
plt.xticks(())
plt.yticks(())

plt.scatter(X0[:,0],X0[:,1],c='#d62728')
plt.scatter(X1[:,0],X1[:,1],c='#9467bd')
plt.scatter(X2[:,0],X2[:,1],c='#17becf')

plt.show()