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

4.Assign the points for representing in the graph.

5.Predict the regression for marks by using the representation of the graph.

6.Compare the graphs and hence we obtained the linear regression for the given datas.

## Program:
```
/*
Program to implement the simple linear regression model for predicting the marks scored.
Developed by: S.Subashini
RegisterNumber:  212222240106
*/
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error,mean_squared_error
import matplotlib.pyplot as plt

dataset=pd.read_csv('student_scores.csv')
print(dataset.head())
print(dataset.tail())

#assigning hours to X & scores to Y
X = dataset.iloc[:,:-1].values
print(X)
Y = dataset.iloc[:,-1].values
print(Y)

from sklearn.model_selection import train_test_split
X_train,X_test,Y_train,Y_test = train_test_split(X,Y,test_size=1/3,random_state=0)
print(X_train)
print(X_test)
print(Y_train)
print(Y_test)

from sklearn.linear_model import LinearRegression
reg=LinearRegression()
reg.fit(X_train,Y_train)
Y_pred =reg.predict(X_test)
print(Y_pred)
print(Y_test)

#Graph plot for training data
plt.scatter(X_train,Y_train,color='blue')
plt.plot(X_train,reg.predict(X_train),color='purple')
plt.title("Hours vs Scores(Training set)")
plt.xlabel("Hours")
plt.ylabel("Scores")
plt.show()

#Graph plot for test data
plt.scatter(X_test,Y_test,color='red')
plt.plot(X_train,reg.predict(X_train),color='purple')
plt.title("Hours vs Scores(Testing set)")
plt.xlabel("Hours")
plt.ylabel("Scores")
plt.show()

mse=mean_absolute_error(Y_test,Y_pred)
print('MSE = ',mse)
mae=mean_absolute_error(Y_test,Y_pred)
print('MAE = ',mae)
rmse=np.sqrt(mse)
print("RMSE= ",rmse)
```

## Output:
![simple linear regression model for predicting the marks scored](sam.png)

## 1. df.head()
![image](https://github.com/SubashiniSenniappan/Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored/assets/119404951/17850d51-ffc6-4317-a0e9-32940df803c7)

## 2. df.tail()
![262950286-18485cf5-edeb-47aa-8681-89519d735c72](https://github.com/SubashiniSenniappan/Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored/assets/119404951/e6a3d4a0-205d-4e0b-acf7-67d1060773dd)

## 3. Array value of X
![262950328-8f8b89cc-7dc2-4c94-89a9-c7a1ef2e8313](https://github.com/SubashiniSenniappan/Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored/assets/119404951/e97c94f3-4d56-4956-8e58-acae2689fb39)
### 4. Array value of Y

![262950432-670cdc76-9c84-4ef9-a007-7a823beb9803](https://github.com/SubashiniSenniappan/Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored/assets/119404951/e4161ddf-ba23-4c7d-bf26-d1c59a81aae8)
### 5. Values of Y prediction
![image](https://github.com/SubashiniSenniappan/Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored/assets/119404951/0c582493-f29f-4b1b-ab4f-f5788d6f848a)
## 6. Array values of Y test
![image](https://github.com/SubashiniSenniappan/Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored/assets/119404951/d002dfa9-4d65-4ade-82c5-0881afb17237)
## 7. Training Set Graph
![image](https://github.com/SubashiniSenniappan/Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored/assets/119404951/5e56f384-5409-459a-82f3-d033a1a432a5)
## 8. Test Set Graph
![image](https://github.com/SubashiniSenniappan/Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored/assets/119404951/3cc2057f-6591-4ac1-ba3d-231babe1aa7d)
## 9. Values of MSE, MAE and RMSE

![image](https://github.com/SubashiniSenniappan/Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored/assets/119404951/e3f82eb9-bdf1-486f-85dd-06a2add40a7f)



## Result:
Thus the program to implement the simple linear regression model for predicting the marks scored is written and verified using python programming.
