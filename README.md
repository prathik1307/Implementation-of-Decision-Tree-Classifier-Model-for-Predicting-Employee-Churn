 # Implementation-of-Decision-Tree-Classifier-Model-for-Predicting-Employee-Churn

## AIM:
To write a program to implement the Decision Tree Classifier Model for Predicting Employee Churn.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
```
1.Import pandas
2.Import Decision tree classifier
3.Fit the data in the model
4.Find the accuracy score 
```
## Program:
```
/*
Program to implement the Decision Tree Classifier Model for Predicting Employee Churn.
Developed by:PRATHIKSHA R
RegisterNumber:  212224040244
import pandas as pd
data = pd.read_csv("/content/Employee.csv")
data.head()
print(data.head())

data.info()
data.isnull().sum()
data["left"].value_counts()

print(data.head())

from sklearn.preprocessing import LabelEncoder
le=LabelEncoder()
data["salary"] = le.fit_transform(data["salary"])


x = data[["satisfaction_level","last_evaluation","number_project","average_montly_hours","time_spend_company","Work_accident","promotion_last_5years",
         "salary"]]
print(x.head())


y = data["left"]

from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.3,random_state=0)

from sklearn.tree import DecisionTreeClassifier
dt = DecisionTreeClassifier(criterion="entropy")
dt.fit(x_train,y_train)
y_pred = dt.predict(x_test)

from sklearn import metrics
accuracy = metrics.accuracy_score(y_test,y_pred)
print(accuracy)

dt.predict([[0.5,0.8,9,260,6,0,1,2]])
import matplotlib.pyplot as plt
from sklearn.tree import plot_tree
plt.figure(figsize=(6,8))
plot_tree(dt,feature_names=x.columns,class_names=['salary','left'],filled=True)

plt.show()

*/

```

## Output:

![image](https://github.com/user-attachments/assets/8e45f992-5694-42d4-a935-085d75395ec0)
![image](https://github.com/user-attachments/assets/76ce7c3c-777b-4658-933a-05088c668d8b)
![image](https://github.com/user-attachments/assets/2d1dbf86-a870-4997-a1df-67888f793dfb)
![image](https://github.com/user-attachments/assets/ea1ca667-4cf1-4ca8-9891-747996b61bf1)
![image](https://github.com/user-attachments/assets/95427891-f808-4c4c-95c3-2f6b165ac40f)
![image](https://github.com/user-attachments/assets/3e577f0e-9262-4a69-9834-bf95b3155dd1)
![image](https://github.com/user-attachments/assets/2af58dbb-4a40-440b-a7ee-fdef59c9f78d)
![image](https://github.com/user-attachments/assets/8dac92d3-e855-4102-bc63-2aa75713993f)


## Result:
Thus the program to implement the  Decision Tree Classifier Model for Predicting Employee Churn is written and verified using python programming.
