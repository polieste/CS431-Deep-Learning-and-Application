import numpy as np
import matplotlib.pyplot as plt
import math
import random
from keras.layers import Dense, Input,Flatten
from keras import Model,models,optimizers
import tensorflow as tf
from sklearn import preprocessing

(x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()

print("shape x,y_train:",x_train.shape,y_train.shape)
print("shape x,y_test:",x_test.shape,y_test.shape)
# reshape
X_train = x_train / 255
X_test = x_test / 255
X_train_flattened = X_train.reshape(len(X_train), 28*28)
X_test_flattened = X_test.reshape(len(X_test), 28*28)


class neural_network:
    def __init__(self):
        return None
    def build(self,input_dim,output_dim):

        input=Input((input_dim,))
        hidden_layer1=Dense(100,activation='relu',use_bias=True)(input)
        output=Dense(output_dim,activation='softmax',use_bias=True)(hidden_layer1)
        self.model=Model(input,output)
        return self.model
    def train(self,x_train,y_train):

        self.model.compile(optimizer='Adam',loss='sparse_categorical_crossentropy')

        hist=self.model.fit(x_train,y_train,epochs=10)
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

Neural_network_Model=neural_network()
Neural_network_Model.build(28*28,10)
hist=Neural_network_Model.train(X_train_flattened,y_train)
weights=Neural_network_Model.get_paramater()
#print(weights)

plt.plot(hist.history['loss'])
plt.show()

y_pred=Neural_network_Model.predict(X_test_flattened)
y_pred=[y.argmax() for y in y_pred]
print(y_pred[0:10])
print(y_test[:10])
