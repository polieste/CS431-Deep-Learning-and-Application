import numpy as np
import matplotlib.pyplot as plt
from keras.layers import Dense, Input
from keras import Model,models
import random

#b1 generate data
means = [[1, 2], [2, 3]]
cov = [[1, 0], [0, 1]]
N = 100
X0 = np.random.multivariate_normal(means[0], cov, N)
X1 = np.random.multivariate_normal(means[1], cov, N)
X = np.concatenate((X0, X1), axis = 0)
#X.shape()
Y= np.asarray([0]*N + [1]*N).T
plt.scatter(X0[:,0],X0[:,1],c='#d62728')
plt.scatter(X1[:,0],X1[:,1],c='#9467bd')
plt.show()

#b2 code model vs 8 phuong thuc
class logistic_regression:
    def __init__(self):
        return None
    def build(self,in_dim=2):
        input=Input(in_dim)
        output=Dense(1,activation='sigmoid',use_bias=True)(input)
        self.model=Model(input,output)
        return self.model
    def train(self,x_train,y_train):
        self.model.compile(optimizer='SGD',loss='binary_crossentropy')
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
logisticModel=logistic_regression()
logisticModel.build()
hist=logisticModel.train(X,Y)
weights=logisticModel.get_paramater()
#print(weights)
w0=weights[1][0]
w1=weights[0][0][0]#x
w2=weights[0][1][0]#y
print(w0,w1,w2)
x1=np.arange(0,8,0.5)
x2=-w1/w2 * x1 - w0/w2
plt.scatter(X0[:,0],X0[:,1],c='#d62728')
plt.scatter(X1[:,0],X1[:,1],c='#9467bd')
plt.plot(x1,x2)
plt.show()
#b4 visualize loss+ hyper plane
plt.plot(hist.history['loss'])
plt.show()