from sklearn.svm import SVC
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score

df = pd.read_csv("train.csv")
df = df.dropna()

print(df.head())

df['Sex'] = df['Sex'].map({'male': 0, 'female': 1})
df['Embarked'] = df['Embarked'].map({'S': 0, 'C': 1, 'Q': 2})


X = df['Pclass', 'Sex', 'SibSp', 'Parch', 'Fare', 'Embarked']
y = df['Survived']

clf = SVC.fit(X,y)