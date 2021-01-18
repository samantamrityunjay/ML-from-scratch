import numpy as np
from sklearn.model_selection import train_test_split
from sklearn import datasets
import matplotlib.pyplot as plt 


X,y = datasets.make_classification(n_samples=1000, n_features=10, n_classes=2, random_state=123)
X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.2,random_state=2021)

def accuracy(true,predict):
    return np.sum(true==predict) / len(predict)

from nb import NB
naivebayes = NB()
naivebayes.fit(X_train,y_train)
predict = naivebayes.predict(X_test)
print("Accuracy for classification dataset using NB " , accuracy(y_test,predict))
