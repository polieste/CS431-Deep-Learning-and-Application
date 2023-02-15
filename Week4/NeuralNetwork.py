import numpy as np
import matplotlib.pyplot as plt
from keras.layers import Dense, Input
from keras import Model,models,optimizers
from sklearn.datasets import make_circles
#b1 generate data
X, Y = make_circles(n_samples=5000, factor=0.3, noise=0.05, random_state=0)
print(X.shape,Y.shape)
print(X[0:5],Y[0:5])
plt.scatter(X[:, 0], X[:, 1], c=Y)
plt.ylabel("Feature #1")
plt.xlabel("Feature #0")
plt.title("data")
plt.show()

#b3 sua lai ham build voi 1 lop an, 5 neural
class neural_network:
    def __init__(self):
        return None
    def build(self,in_dim=2):
        input=Input(in_dim)
        hidden_layer=Dense(5,activation='relu',use_bias=True)(input)
        output=Dense(1,activation='sigmoid',use_bias=True)(hidden_layer)
        self.model=Model(input,output)
        return self.model
    def train(self,x_train,y_train):
        otm=optimizers.SGD(learning_rate=0.0001,momentum=0.9)
        self.model.compile(optimizer=otm,loss='binary_crossentropy')
        hist=self.model.fit(x_train,y_train,epochs=3000)
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


#b4 train va visualize loss
Neural_network_Model=neural_network()
Neural_network_Model.build()
hist=Neural_network_Model.train(X,Y)
weights=Neural_network_Model.get_paramater()
print(weights)

plt.plot(hist.history['loss'])
plt.show()


for i in range(0,5):
    w0 = weights[1][i]
    w1 = weights[0][0][i]  # x
    w2 = weights[0][1][i]  # y
    print(w0, w1, w2)
    x = np.arange(X[:,0].min(), X[:,0].max(), 1)
    y = -w1 / w2 * x - w0 / w2
    plt.plot(x,y)

plt.scatter(X[:, 0], X[:, 1], c=Y)
plt.xlim(-1.5, 1.5)
plt.ylim(-1.5, 1.5)
plt.ylabel("Feature #1")
plt.xlabel("Feature #0")
plt.title("data")

plt.show()



#b5 ve decision boundary
xm = np.arange(X[:,0].min(), X[:,0].max(), 0.015)
xlen = len(xm)
ym = np.arange(X[:,1].min(), X[:,1].max(), 0.015)
ylen = len(ym)
xx, yy = np.meshgrid(xm, ym)


xx1 = xx.ravel().reshape(1, xx.size)
yy1 = yy.ravel().reshape(1, yy.size)

print(xx.shape, yy.shape)
print(xx1.shape, yy1.shape)
XX = np.concatenate(( xx1, yy1), axis = 0)
print(XX.shape)
XX=XX.T
print(XX.shape)
Z=Neural_network_Model.predict(XX)
print(Z.shape)
Z1=[]
for z in Z:
    if z<0.5:
        Z1.append(0)
    else:
        Z1.append(1)
Z1=np.array(Z1)
Z1 = Z1.reshape(xx.shape)
print(Z1)

CS = plt.contourf(yy, xx, Z1, 200, cmap='jet', alpha = .1)


#plt.xlim(X[:,0].min(), X[:,0].max())
#plt.ylim(X[:,1].min(), X[:,1].max())
plt.xlim(-1.5, 1.5)
plt.ylim(-1.5, 1.5)
plt.xticks(())
plt.yticks(())

plt.scatter(X[:, 0], X[:, 1], c=Y)
plt.ylabel("Feature #1")
plt.xlabel("Feature #0")
plt.title("data")
plt.show()
