import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

import os
print(os.listdir())

import warnings
warnings.filterwarnings('ignore')

dataset = pd.read_csv("heart.csv")

type(dataset)

dataset.shape

dataset.head(5)

from sklearn.model_selection import train_test_split

predictors = dataset.drop("target",axis=1)
target = dataset["target"]

X_train,X_test,Y_train,Y_test = train_test_split(predictors,target,test_size=0.20,random_state=0)

X_train.shape
X_test.shape
Y_train.shape
Y_test.shape

from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression

lr = LogisticRegression()

lr.fit(X_train,Y_train)

Y_pred_lr = lr.predict(X_test)

Y_pred_lr.shape

score_lr = round(accuracy_score(Y_pred_lr,Y_test)*100,2)

print("The accuracy score achieved using Logistic Regression is: "+str(score_lr)+" %")

# Accuracy Plot
plt.figure(figsize=(8,6))
plt.bar(["Logistic Regression"], [score_lr])
plt.xlabel("Model")
plt.ylabel("Accuracy (%)")
plt.title("Model Accuracy")
plt.show()

# Count Plot
plt.figure(figsize=(8,6))
sns.countplot(x="target", data=dataset)
plt.xlabel("Target")
plt.ylabel("Count")
plt.title("Target Count")
plt.show()