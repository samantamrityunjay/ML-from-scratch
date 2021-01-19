import numpy as np 
from sklearn.model_selection import train_test_split
from sklearn import datasets
import matplotlib.pyplot as plt 


from perceptron import Perceptron

def accuracy(true,predicted):
    return np.sum(true==predicted)/len(true)

X,y = datasets.make_blobs(n_samples=150, centers = 2, random_state = 123)
X_train, X_test, y_train, y_test = train_test_split(X,y,test_size = 0.2, random_state =2021)


p= Perceptron(lr=0.01)
p.fit(X_train,y_train)
predictions = p.predict(X_test)

print("Perceptron classification accuracy ",accuracy(y_test,predictions))