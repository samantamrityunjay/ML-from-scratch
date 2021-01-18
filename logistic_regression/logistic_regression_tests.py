import numpy as np
from sklearn.model_selection import train_test_split
from sklearn import datasets
import matplotlib.pyplot as plt 


bc = datasets.load_breast_cancer()
X,y = bc.data, bc.target
X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.2,random_state=2021)

def accuracy(true,predict):
    return np.sum(true==predict) / len(predict)

from logistic_regression import LogisticRegression
regressor = LogisticRegression(lr=0.0001)
regressor.fit(X_train,y_train)
predict = regressor.predict(X_test)
print("Accuracy for breast cancer dataset using LR " , accuracy(y_test,predict))
