# Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student

## AIM:
To write a program to implement the the Logistic Regression Model to Predict the Placement Status of Student.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1.Import libraries & load data using pandas, and preview with df.head().
2.Clean data by dropping sl_no and salary, checking for nulls and duplicates.
3.Encode categorical columns (like gender, education streams) using LabelEncoder.
4.Split features and target:
X = all columns except status
y = status (Placed/Not Placed)
5.Train-test split (80/20) and initialize LogisticRegression.
6.Fit the model and make predictions.
7.Evaluate model with accuracy, confusion matrix, and classification report.

## Program:
```
/*
Program to implement the the Logistic Regression Model to Predict the Placement Status of Student.
Developed by  : VIKASKUMAR M 
RegisterNumber: 212224220122
*/
```
```
import pandas as pd
data= pd.read_csv('/content/Placement_Data.csv')
data.head()
```
```
data1=data.copy()
data1=data.drop(['sl_no','salary'],axis=1)
data1.head()
```
```
data1.isnull().sum()
```
```
data1.duplicated().sum()
```
```
from sklearn.preprocessing import LabelEncoder
le=LabelEncoder()
data1["gender"]=le.fit_transform(data1["gender"])
data1["ssc_b"]=le.fit_transform(data1["ssc_b"])
data1["hsc_b"]=le.fit_transform(data1["hsc_b"])
data1["hsc_s"]=le.fit_transform(data1["hsc_s"])
data1["degree_t"]=le.fit_transform(data1["degree_t"])
data1["workex"]=le.fit_transform(data1["workex"])
data1["specialisation"]=le.fit_transform(data1["specialisation"])
data1["status"]=le.fit_transform(data1["status"])
data1
```
```
x=data1.iloc[:,: -1]
x
```
```
y=data1["status"]
y
```
```
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=0)

from sklearn.linear_model import LogisticRegression
model=LogisticRegression(solver="liblinear")
model.fit(x_train,y_train)
y_pred=model.predict(x_test)
```
```
from sklearn.metrics import accuracy_score,confusion_matrix,classification_report
accuracy=accuracy_score(y_test,y_pred)
confusion=confusion_matrix(y_test,y_pred)
cr=classification_report(y_test,y_pred)
print("Accuracy score:",accuracy)
print("\nConfusion matrix:\n",confusion)
print("\nClassification Report:\n",cr)
```

## Output:
![image](https://github.com/user-attachments/assets/76f46377-ce30-45b2-9102-921ae50d3186)

![image](https://github.com/user-attachments/assets/337ca214-ae59-4b96-a9a7-a73db5be2a2c)

![image](https://github.com/user-attachments/assets/81a37966-78ed-4965-bf1c-8f53a79f63c0)

![image](https://github.com/user-attachments/assets/aadb3a5a-d288-49e9-a166-26ad67a3613b)

![image](https://github.com/user-attachments/assets/3a6a26e3-c2d8-41eb-b636-ba4399a7c629)

![image](https://github.com/user-attachments/assets/500691e8-350c-49df-9090-b19be7fa91ee)

![image](https://github.com/user-attachments/assets/4c01da79-263b-463d-af65-de06a6df65e9)

![image](https://github.com/user-attachments/assets/fb5bd65c-9f5f-4c05-ac67-f4a1de4c8c7a)

## Result:
Thus the program to implement the the Logistic Regression Model to Predict the Placement Status of Student is written and verified using python programming.
