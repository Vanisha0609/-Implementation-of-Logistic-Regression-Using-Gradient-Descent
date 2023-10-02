# Implementation-of-Logistic-Regression-Using-Gradient-Descent

## AIM:
To write a program to implement the the Logistic Regression Using Gradient Descent.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1. Use the standard libraries in python for finding linear regression
2. Set variables for assigning dataset values.
3. Import linear regression from sklearn.
4. Predict the values of array.
5. Calculate the accuracy, confusion and classification report by importing the required modules from sklearn.
6. Obtain the graph

## Program:
```
/*
Program to implement the the Logistic Regression Using Gradient Descent.
Developed by: Vanisha Ramesh
RegisterNumber:  212222040174
*/
```
```
import numpy as np
import matplotlib.pyplot as plt
from scipy import optimize

data=np.loadtxt("/content/ex2data1.txt",delimiter=',')
x=data[:,[0,1]]
y=data[:,2]

x[:5]

y[:5]

plt.figure()
plt.scatter(x[y==1][:,0],x[y==1][:,1],label="Admitted",color="cadetblue")
plt.scatter(x[y==0][:,0],x[y==0][:,1],label="Not Admitted",color="plum")
plt.xlabel("Exam 1 score")
plt.ylabel("Exam 2 score")
plt.legend()
plt.show()

def sigmoid(z):
  return 1/(1+np.exp(-z))

plt.plot()
x_plot=np.linspace(-10,10,100)
plt.plot(x_plot,sigmoid(x_plot),color="cadetblue")
plt.show()

def costFunction(theta,x,y):
  h=sigmoid(np.dot(x,theta))
  j=-(np.dot(y,np.log(h))+np.dot(1-y,np.log(1-h)))/x.shape[0]
  grad=np.dot(x.T,h-y)/x.shape[0]
  return j,grad

x_train=np.hstack((np.ones((x.shape[0],1)),x))
theta=np.array([0,0,0])
j,grad=costFunction(theta,x_train,y)
print(j)
print(grad)

x_train=np.hstack((np.ones((x.shape[0],1)),x))
theta=np.array([-24,0.2,0.2])
j,grad=costFunction(theta,x_train,y)
print(j)
print(grad)

def cost(theta,x,y):
  h=sigmoid(np.dot(x,theta))
  j= -(np.dot(y, np.log(h)) + np.dot(1-y, np.log(1-h)))/x.shape[0]
  return j


def gradient(theta,x,y):

  h=sigmoid(np.dot(x,theta))
  grad=np.dot(x.T,h-y)/x.shape[0]
  return grad


x_train=np.hstack((np.ones((x.shape[0],1)),x))
theta=np.array([0,0,0])
res=optimize.minimize(fun=cost,x0=theta,args=(x_train,y),method='Newton-CG',jac=gradient)
print(res.fun)
print(res.x)

def plotDecisionBoundary(theta,x,y):
  x_min,x_max=x[:,0].min()-1,x[:,0].max()+1
  y_min,y_max=x[:,1].min()-1,x[:,1].max()+1
  xx,yy=np.meshgrid(np.arange(x_min,x_max,0.1),np.arange(y_min,y_max,0.1))
  x_plot=np.c_[xx.ravel(),yy.ravel()]
  x_plot=np.hstack((np.ones((x_plot.shape[0],1)),x_plot))
  y_plot=np.dot(x_plot,theta).reshape(xx.shape)

  plt.figure()
  plt.scatter(x[y==1][:,0],x[y==1][:,1],label="Admitted",color="mediumpurple")
  plt.scatter(x[y==0][:,0],x[y==0][:,1],label="Not admitted",color="pink")
  plt.contour(xx,yy,y_plot,levels=[0])
  plt.xlabel("Exam 1 score")
  plt.ylabel("Exam 2 score")
  plt.legend()
  plt.show()

plotDecisionBoundary(res.x,x,y)

prob=sigmoid(np.dot(np.array([1,45,85]),res.x))
print(prob)

def predict(theta,x):
  x_train=np.hstack((np.ones((x.shape[0],1)),x))
  prob=sigmoid(np.dot(x_train,theta))
  return(prob>=0.5).astype(int)

np.mean(predict(res.x,x)==y)
```

## Output:
1.Array Value of x

![image](https://github.com/Vanisha0609/-Implementation-of-Logistic-Regression-Using-Gradient-Descent/assets/119104009/7b703505-f236-4c80-8988-d96ef8e925b2)

2.Array Value of y

![image](https://github.com/Vanisha0609/-Implementation-of-Logistic-Regression-Using-Gradient-Descent/assets/119104009/7f889494-0d19-4cec-90eb-d164378cdb78)

3.Exam 1-Score Graph

![image](https://github.com/Vanisha0609/-Implementation-of-Logistic-Regression-Using-Gradient-Descent/assets/119104009/f2e06d2d-aa5a-4598-a734-88b4efadd634)

4.Sigmoid function graph

![image](https://github.com/Vanisha0609/-Implementation-of-Logistic-Regression-Using-Gradient-Descent/assets/119104009/d2f2d93c-17bd-4f76-ac6a-27bc817d9b9c)

5.x_train_grad value

![image](https://github.com/Vanisha0609/-Implementation-of-Logistic-Regression-Using-Gradient-Descent/assets/119104009/60304fa9-b41a-4728-a502-c6a02cb079ce)

6.y_train_grad value

![image](https://github.com/Vanisha0609/-Implementation-of-Logistic-Regression-Using-Gradient-Descent/assets/119104009/a23f1015-eed6-41f6-a731-9676ca6463cd)

7.Print res.x

![image](https://github.com/Vanisha0609/-Implementation-of-Logistic-Regression-Using-Gradient-Descent/assets/119104009/c9abbde0-3b9d-4c95-84f0-951a8bd8f5dd)

8.Decision boundary-graph for exam score

![image](https://github.com/Vanisha0609/-Implementation-of-Logistic-Regression-Using-Gradient-Descent/assets/119104009/520b9622-c640-436d-8f67-5ac5b71bdce6)

9.Probability value

![image](https://github.com/Vanisha0609/-Implementation-of-Logistic-Regression-Using-Gradient-Descent/assets/119104009/47bcbe66-97e9-457b-b512-1d49a2f63f3f)

10.Prediction value of mean

![image](https://github.com/Vanisha0609/-Implementation-of-Logistic-Regression-Using-Gradient-Descent/assets/119104009/058279f9-1050-4c3d-8897-8d4565fcd111)


## Result:
Thus the program to implement the the Logistic Regression Using Gradient Descent is written and verified using python programming.

