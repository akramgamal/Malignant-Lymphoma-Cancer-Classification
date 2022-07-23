import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, recall_score, precision_score
import os
from sklearn.neighbors import KNeighborsClassifier
import cv2
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
import tensorflow as tf
from keras.optimizers import Adam, RMSprop
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense,Dropout,Activation,Flatten,Conv2D,MaxPooling2D
from sklearn.metrics import classification_report
cll='/Users/Zamzam/Desktop/GP/CLL'
fl='/Users/Zamzam/Desktop/GP/FL'
mcl='/Users/Zamzam/Desktop/GP/MCL'
x=[]
y=[]
for j in os.listdir(cll):
    img=cv2.imread(cll+'/'+j)
    img =cv2.resize(img,(224,224))
    x.append(img)
    y.append(0)
for j in os.listdir(fl):
    img=cv2.imread(fl+'/'+j)
    img =cv2.resize(img,(224,224))
    x.append(img)
    y.append(1)
for j in os.listdir(mcl):
    img=cv2.imread(mcl+'/'+j)
    img =cv2.resize(img,(224,224))
    x.append(img)
    y.append(2)

# val1=[]
# val2=[]
# cll='/Users/Zamzam/Desktop/GP/val/valCLL'
# fl='/Users/Zamzam/Desktop/GP/val/valFL'
# mcl='/Users/Zamzam/Desktop/GP/val/valMCL'
# for j in os.listdir(cll):
#     img=cv2.imread(cll+'/'+j)
#     img =cv2.resize(img,(224,224))
#     x.append(img)
#     y.append(0)
# for j in os.listdir(fl):
#     img=cv2.imread(fl+'/'+j)
#     img =cv2.resize(img,(224,224))
#     x.append(img)
#     y.append(1)
# for j in os.listdir(mcl):
#     img=cv2.imread(mcl+'/'+j)
#     img =cv2.resize(img,(224,224))
#     x.append(img)
#     y.append(2)

# test1=[]
# test2=[]
# cll='/Users/Zamzam/Desktop/GP/test/testCLL'
# fl='/Users/Zamzam/Desktop/GP/test/testFL'
# mcl='/Users/Zamzam/Desktop/GP/test/testMCL'
# for j in os.listdir(cll):
#     img=cv2.imread(cll+'/'+j)
#     img =cv2.resize(img,(224,224))
#     x.append(img)
#     y.append(0)
# for j in os.listdir(fl):
#     img=cv2.imread(fl+'/'+j)
#     img =cv2.resize(img,(224,224))
#     x.append(img)
#     y.append(1)
# for j in os.listdir(mcl):
#     img=cv2.imread(mcl+'/'+j)
#     img =cv2.resize(img,(224,224))
#     x.append(img)
#     y.append(2)
print(len(x))
x=np.array(x)
y=np.array(y)
print(x.shape)
x=x.reshape(len(x),-1)
print(x.shape)
x=x/255.0

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.20,shuffle=True)

lg=LogisticRegression()
lg.fit(x_train,y_train)
y_pred = lg.predict(x_test)
print(classification_report(y_test, y_pred))
print("train accuracy for logistic Regression ",lg.score(x_train,y_train))
print("test accuracy for logistic Regression",lg.score(x_test,y_test)) 


dt=DecisionTreeClassifier()
dt.fit(x_train,y_train)
y_pred = dt.predict(x_test)
print(classification_report(y_test, y_pred))
print("train accuracy for Decision tree",dt.score(x_train,y_train))
print("test accuracy for Decision tree",dt.score(x_test,y_test)) 

#kernel='linear'

sv=SVC(kernel='linear')
sv.fit(x_train,y_train)
y_pred =sv.predict(x_test)
print(classification_report(y_test, y_pred))
print("train accuracy for svm",sv.score(x_train,y_train)) 
print("test accuracy for svm",sv.score(x_test,y_test)) 

knn = KNeighborsClassifier(n_neighbors=9)
knn.fit(x_train,y_train)
y_pred = knn.predict(x_test)
print(classification_report(y_test, y_pred))
print("train accuracy for KNN",knn.score(x_train,y_train)) 
print("test accuracy for KNN",knn.score(x_test,y_test)) 
