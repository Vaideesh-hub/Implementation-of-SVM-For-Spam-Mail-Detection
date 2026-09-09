# Implementation-of-SVM-For-Spam-Mail-Detection

## AIM:
To write a program to implement the SVM For Spam Mail Detection.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
 1.Import the spam dataset, detect file encoding, and separate message text and labels.
 2.Use CountVectorizer to transform text messages into numerical vectors.
 3.Split the dataset into training and testing sets and train the Support Vector Machine classifier.
 4.Predict spam/ham messages, calculate accuracy score, and generate the classification report.

## Program:
```
/*
Program to implement the SVM For Spam Mail Detection..
Developed by: MIRDULA D
RegisterNumber:  212225040234
*/
```
```
import pandas as pd
import chardet
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.svm import SVC

from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix
)

file = "spam.csv"

with open(file, 'rb') as rawdata:
    result = chardet.detect(rawdata.read(100000))

data = pd.read_csv(file, encoding=result['encoding'])

print(data.head())

x = data["v2"].values
y = data["v1"].values

x_train, x_test, y_train, y_test = train_test_split(
    x,
    y,
    test_size=0.2,
    random_state=42
)

cv = CountVectorizer()

x_train = cv.fit_transform(x_train)
x_test = cv.transform(x_test)

svc = SVC()

svc.fit(x_train, y_train)

y_pred = svc.predict(x_test)

print("\nPREDICTED VALUES:\n")
print(y_pred[:10])

accuracy = accuracy_score(y_test, y_pred)

print("\nACCURACY SCORE:")
print(accuracy)

report = classification_report(y_test, y_pred)

print("\nCLASSIFICATION REPORT:\n")
print(report)

cm = confusion_matrix(y_test, y_pred)

print("\nCONFUSION MATRIX:\n")
print(cm)

plt.figure(figsize=(6,4))

sns.heatmap(
    cm,
    annot=True,
    fmt='d',
    cmap='Blues',
    xticklabels=['ham', 'spam'],
    yticklabels=['ham', 'spam']
)

plt.xlabel("Predicted Label")
plt.ylabel("True Label")
plt.title("Confusion Matrix")

plt.show()

new_message = ["Congratulations! You won a free iPhone"]

new_message_vector = cv.transform(new_message)

prediction = svc.predict(new_message_vector)

print("\nNEW MESSAGE PREDICTION:")
print(prediction)
```

## Output:
<img width="904" height="557" alt="597450092-00f91233-9b03-4fb9-9f2e-034baee5ba21" src="https://github.com/user-attachments/assets/6ec7b12f-49ab-494b-8d6d-e121e374b77d" />


<img width="905" height="527" alt="597450114-56117dee-4ffc-4343-8ee9-4a228e0e82a3" src="https://github.com/user-attachments/assets/2c29fb4c-8825-40ea-852a-1f83fd1c900b" />


## Result:
Thus the program to implement the SVM For Spam Mail Detection is written and verified using python programming.
