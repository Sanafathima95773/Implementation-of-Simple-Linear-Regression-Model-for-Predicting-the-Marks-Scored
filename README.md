# Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored

## AIM:
To write a program to predict the marks scored by a student using the simple linear regression model.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1.Import the standard Libraries.

2.Set variables for assigning dataset values.

3.Import linear regression from sklearn.

4.Assign the points for representing in the graph

5.Predict the regression for marks by using the representation of the graph.

6.Compare the graphs and hence we obtained the linear regression for the given datas.

## Program:
```
/*
Implementation of Univariate Linear Regression
AIM:
To implement univariate Linear Regression to fit a straight line using least squares.

Equipments Required:
Hardware – PCs
Anaconda – Python 3.7 Installation / Jupyter notebook
Algorithm
Get the independent variable X and dependent variable Y.
Calculate the mean of the X -values and the mean of the Y -values.
Find the slope m of the line of best fit using the formula.
image

4. Compute the y -intercept of the line by using the formula:
image

5. Use the slope m and the y -intercept to form the equation of the line. 6. Obtain the straight line equation Y=mX+b and plot the scatterplot.

Program:

```
/*
Program to implement the simple linear regression model for predicting the marks scored.
Developed by: Sana Fathima H
RegisterNumber: 212223240145
*/
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
df=pd.read_csv('/content/MLSET.csv')
df.head(10)
plt.scatter(df['X'],df['Y'])
plt.xlabel('X')
plt.ylabel('Y')
x=df.iloc[:,0:-1]
y=df.iloc[:,-1]
from sklearn.model_selection import train_test_split
X_train,X_test,Y_train,Y_test=train_test_split(x,y,test_size=0.2,random_state=0)
from sklearn.linear_model import LinearRegression
lr=LinearRegression()
lr.fit(X_train,Y_train)
X_train
Y_train
lr.predict(x_test.iloc[0].values.reshape(1,1))
plt.scatter(df['X'],df['Y'])
plt.xlabel('X')
plt.ylabel('Y')
plt.plot(X_train,lr.predict(X_train),color='orange')
lr.coef_
lr.intercept_
```
## Output:
1) Head:
![image](https://github.com/Sanafathima95773/Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored/assets/147084627/487073a2-148b-48e1-99a0-25ed14b87ffc)
2) Graph Of Plotted Data:
![image](https://github.com/Sanafathima95773/Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored/assets/147084627/ba5f8f03-9205-4a13-8eeb-200e370caa36)
3) Trained Data:
![image](https://github.com/Sanafathima95773/Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored/assets/147084627/1bf9af67-cf12-4f8b-b8c0-98e176830e81)
4) Line Of Regression:
![image](https://github.com/Sanafathima95773/Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored/assets/147084627/d3c3d2d1-dc6e-4f6a-81d7-82e1ae60a756)
5) Coefficient And Intercept Values:
![image](https://github.com/Sanafathima95773/Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored/assets/147084627/28df51fd-caae-494e-bd14-dfc1434fb516)

## Result:
Thus the program to implement the simple linear regression model for predicting the marks scored is written and verified using python programming.
