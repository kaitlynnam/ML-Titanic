from sklearn import svm
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.inspection import DecisionBoundaryDisplay
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import GridSearchCV

df = pd.read_csv("train.csv")
df = df.dropna()

#print(df.head())

df['Sex'] = df['Sex'].map({'male': 0, 'female': 1})
df['Embarked'] = df['Embarked'].map({'S': 0, 'C': 1, 'Q': 2})


X = df[['Pclass', 'Sex', 'SibSp', 'Parch', 'Fare', 'Embarked', 'Age']]
y = df['Survived']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

#scaler = StandardScaler()

#X_scaled = scaler.fit_transform(X)
#X_train_scaled = scaler.fit_transform(X_train)
#X_test_scaled = scaler.transform(X_test)
pipeline = make_pipeline(StandardScaler(), svm.SVC(kernel='rbf', C=10, gamma=0.01))

param_grid = {
    'svc__kernel': ['rbf', 'linear'],       # Try both kernels
    'svc__C': [0.1, 1, 10],                 # Regularization strength
    'svc__gamma': ['scale', 0.01, 0.1]      # How far influence of points spreads (for RBF)
}

grid = GridSearchCV(pipeline, param_grid, cv=5)  # 5-fold cross-validation
grid.fit(X_train, y_train)

print("Best Parameters:", grid.best_params_)
print("Best CV Score:", grid.best_score_)


pipeline.fit(X_train, y_train)

accuracy = pipeline.score(X_test, y_test)

#svm_model = svm.SVC(kernel='linear')

#svm_model.fit(X_train_scaled, y_train)

#accuracy = svm_model.score(X_test_scaled, y_test)

#scores = cross_val_score(svm_model, X_scaled, y ,cv=10)
#scores = cross_val_score(pipeline, X, y, cv=10)
#print("Average CV accuracy:", scores.mean())

print(f"Accuracy: {accuracy:.4f}")

y_pred = pipeline.predict(X_test)

print(classification_report(y_test, y_pred))
print(confusion_matrix(y_test,y_pred))


X_2D = X[['Fare', 'Age']].dropna()
y_2D = y[X_2D.index]

scaler = StandardScaler()
X_2D_scaled = scaler.fit_transform(X_2D)

clf = svm.SVC(kernel='rbf')
clf.fit(X_2D_scaled, y_2D)

_, ax = plt.subplots()
DecisionBoundaryDisplay.from_estimator(clf, X_2D_scaled, ax=ax, cmap=plt.cm.Paired)
ax.scatter(X_2D_scaled[:, 0], X_2D_scaled[:, 1], c=y_2D, edgecolors='k')
plt.xlabel('Fare')
plt.ylabel('Age')
plt.show()