import sklearn
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
import pandas as pd
import matplotlib.pyplot as plt


df = pd.read_csv("train.csv")

df = df.dropna()

df['Sex'] = df['Sex'].map({'male': 0, 'female': 1})

df['Embarked'] = df['Embarked'].map({'S': 0, 'C': 1, 'Q': 2})

X = df[['Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Fare', 'Embarked']]

y = df['Survived']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=0)

dt = DecisionTreeClassifier(random_state=0, max_depth=5, min_samples_split=10, min_samples_leaf=2)
dt.fit(X_train, y_train)

accuracy = dt.score(X_test, y_test)

sklearn.tree.plot_tree(dt, filled = True, feature_names=X.columns, class_names=["Dead", "Survived"])
plt.savefig("titanic-tree.png", dpi=300)
plt.show()
plt.close()

train_accuracy = dt.score(X_train, y_train)
test_accuracy = dt.score(X_test, y_test)

print(f"Train Accuracy: {train_accuracy:.4f}")
print(f"Test Accuracy: {test_accuracy:.4f}")