# Ex-02 Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored
## Date:


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
![277167935-fa128769-a5ee-4ab8-96be-36d9989dfbc1](https://github.com/SubashiniSenniappan/Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored/assets/119404951/fa640f3f-fa6d-4cdd-966a-f8c6700db184)

![277167995-e446e77c-4082-4572-a2f6-ae96cc2fe6fd](https://github.com/SubashiniSenniappan/Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored/assets/119404951/be195996-4c94-4d1b-ac6a-dba15cfb3465)




![277168020-7f7796b8-044f-4408-9c48-11af284376c4](https://github.com/SubashiniSenniappan/Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored/assets/119404951/d69b6641-d490-4bdb-964b-ea4ce8659f8e)
04951/c13e0fdd-9df2-466c-be7c-208531f98409)
![277168036-06c589a2-9a2c-4f24-9ca8-6dce58aef77f](https://github.com/SubashiniSenniappan/Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored/assets/119404951/f05d272b-b501-4a5e-bfe8-a26a41deeb8f)

![277168103-3cbcd2ac-9381-475e-a219-aaf29b483eb2](https://github.com/SubashiniSenniappan/Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored/assets/119404951/50224363-1aaf-44f6-9278-304f9d841379)

Thus the program to implement the simple linear regression model for predicting the marks scored is written and verified using python programming.
