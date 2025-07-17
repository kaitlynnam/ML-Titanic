import pandas as pd
import sklearn
from sklearn.inspection import DecisionBoundaryDisplay
from sklearn.linear_model import LogisticRegression
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split


train_df = pd.read_csv("train.csv")

train_df = train_df.dropna()

train_df['Sex'] = train_df['Sex'].map({'female': 0, 'male': 1})

train_df['Embarked'] = train_df['Embarked'].map({'S': 0, 'C': 1, 'Q': 2})

X = train_df[['Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Fare', 'Embarked']]
y = train_df[['Survived']]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)



logreg = LogisticRegression(C=1e5, max_iter=1000)
logreg.fit(X_train, y_train)



y_pred = logreg.predict(X_test)



'''y_pred = pd.Series(y_pred)
y_pred = y_pred.map({1: 'Survived', 0: 'Dead'})'''

acuracy = logreg.score(X_test, y_test)

print(f"Accuracy: {acuracy * 100:.2f}%")

from sklearn.metrics import classification_report, confusion_matrix

print(confusion_matrix(y_test, y_pred))
print(classification_report(y_test, y_pred))

#print(y_pred)

'''y_pred = y_pred.map({'1': 'Survived', '0': 'Dead'})
print(y_pred)'''


'''corr = X.corr()
sns.heatmap(corr[['Survived']], annot=True, cmap='coolwarm')
plt.savefig("heatmap.png")
plt.show()'''

'''sns.pairplot(train_df, hue='Survived')
plt.savefig('pairplot.png')
'''
'''sns.countplot(x="Sex", hue="Survived", data=train_df,)'''

#plt.scatter(X.iloc[:, 1], y.iloc[:, ], c=y.values.ravel(), edgecolors="k", cmap=plt.cm.Paired)

'''plt.hist(train_df[train_df['Survived'] == 1]['Age'].dropna(), bins=30, alpha=0.5, label='Survived', edgecolor='black')
plt.hist(train_df[train_df['Survived'] == 0]['Age'].dropna(), bins=30, alpha=0.5, label='Did not survive', edgecolor='black')'''

'''plt.xlabel('Age')
plt.ylabel('Frequency')
plt.title('Age Distribution by Survival')'''
'''plt.legend()
plt.show()'''

'''plt.xlabel("Pclass")
plt.ylabel("Survived?")

plt.xticks(())
plt.yticks(())
plt.show()'''