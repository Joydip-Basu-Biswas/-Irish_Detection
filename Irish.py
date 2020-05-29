#Iris Data Set

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Importing the data sets
df = pd.read_csv("C:/Users/user/Desktop/Macine_Learning Data set/Iris.csv")
print(df.head())
print(df.info())

print(df["Species"].unique())

# Relation between Fetures and Species
plt.xlabel("Fetures")
plt.ylabel("Species")

plt_x = df.loc[:, "SepalLengthCm"]
plt_y = df.loc[:, "Species"]
plt.scatter(plt_x, plt_y, color="yellow", Label="SepalLengthCm")

plt_x = df.loc[:, "SepalWidthCm"]
plt_y = df.loc[:, "Species"]
plt.scatter(plt_x, plt_y, color="red", Label="SepalWidthCm")

plt_x = df.loc[:, "PetalLengthCm"]
plt_y = df.loc[:, "Species"]
plt.scatter(plt_x, plt_y, color="blue", Label="PetalLengthCm")

plt_x = df.loc[:, "PetalWidthCm"]
plt_y = df.loc[:, "Species"]
plt.scatter(plt_x, plt_y, color="black", Label="PetalWidthCm")

plt.legend(loc=4, prop={"size": 7})
plt.show()

# Divide into x and y
x = df[["SepalLengthCm", "SepalWidthCm", "PetalLengthCm", "PetalWidthCm"]]
y = df[["Species"]]

# Splitting The Data Set into Training Set And Test Set
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=42)


# Feature Scaling
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
x_train = sc.fit_transform(x_train)
x_test = sc.transform(x_test)


# Fitting Logistic Regression to our training Set
from sklearn.linear_model import LogisticRegression

classifier = LogisticRegression(random_state=0)
classifier.fit(x_train, y_train)

# Predicting the test set result
y_pred = classifier.predict(x_test)
print(y_pred)


# Accuracy of Model
from sklearn.metrics import accuracy_score
acc = accuracy_score(y_test, y_pred)
print(acc)
