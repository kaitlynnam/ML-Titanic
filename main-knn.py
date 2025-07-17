from sklearn.neighbors import KNeighborsClassifier
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.inspection import DecisionBoundaryDisplay
from sklearn.model_selection import cross_val_score
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.preprocessing import StandardScaler
from matplotlib.lines import Line2D

scaler = StandardScaler()

df = pd.read_csv("train.csv")

df = df.dropna()

df['Sex'] = df['Sex'].map({'male': 0, 'female': 1})
df['Embarked'] = df['Embarked'].map({'S': 0, 'C': 1, 'Q': 2})

X = df[['Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Fare', 'Embarked']]

y= df['Survived']



X_plot = X[['Age', 'Fare']] 

X_train, X_test, y_train, y_test = train_test_split(X_plot, y, test_size=0.2, random_state=0)


X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

highest_accuracy = 0
best_k = 0

for k in range (1, 30):

    knn = KNeighborsClassifier(n_neighbors=k, algorithm='auto')

    knn.fit(X_train_scaled, y_train)

    accuracy = knn.score(X_test_scaled, y_test)

    if accuracy > highest_accuracy:
        highest_accuracy = accuracy
        best_k = k

    print(f"The accuracy for k = {k} is: {accuracy:.4f} ")

print(f"The best k is {best_k} with an accuracy of {accuracy:.4f}")

knn = KNeighborsClassifier(n_neighbors=best_k, algorithm='auto')
knn.fit(X_train_scaled, y_train)


scores = cross_val_score(knn, scaler.transform(X_plot), y, cv=10)  # 10-fold CV
print("Average CV accuracy:", scores.mean())


_, ax = plt.subplots()



DecisionBoundaryDisplay.from_estimator(knn, scaler.transform(X_plot), cmap=plt.cm.Paired, ax=ax, response_method="predict", plot_method="pcolormesh", shading="auto", xlabel="Age", ylabel="Fare", eps=0.5)
scatter = ax.scatter(X_train_scaled[:, 0], X_train_scaled[:, 1], c=y_train)




plt.xlabel('Age')
plt.ylabel('Fare')

plt.savefig('scatter.png')
#all blue because of a class imballence, it is far more probable to the algorithm that some one is dead rather than alive
plt.show()
plt.close()

y_pred = knn.predict(X_test_scaled)

print("Confusion Matrix:")
print(confusion_matrix(y_test, y_pred))

print("\nClassification Report:")
print(classification_report(y_test, y_pred))