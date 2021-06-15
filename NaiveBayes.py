import pandas as pd
import numpy as np
import io
import requests
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import classification_report, accuracy_score

dataset = pd.read_csv("dataset.csv")
#print("1st 10 dataset sample:-\n", dataset.head(10))

X = dataset.iloc[:, 0:4].values
y = dataset.iloc[:, 4].values
#print(X)
#print(y)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)

sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

model = GaussianNB()
model.fit(X_train, y_train)
y_pred = model.predict(X_test)

print("\nPrediction for Test Data is:-\n",y_pred)
print("\nClassification Report is:-\n",classification_report(y_test,y_pred))
print("\nAccuracy of Prediction is:-",accuracy_score(y_test,y_pred)*100)
