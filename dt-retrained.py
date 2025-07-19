import sklearn
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix

df = pd.read_csv("train.csv")

df = df.dropna()

df['Sex'] = df['Sex'].map({'male': 0, 'female': 1})

df['Title'] = (df['Name'].astype(str).str.split(",", n=1).str[1].str.split(".", n=1).str[0].str.strip()
)                    
title_counts = df['Title'].value_counts().sort_values(ascending=False)

title_survival = df.groupby("Title")["Survived"].mean().sort_values()

df["FamilySize"] = df["SibSp"] + df["Parch"] + 1

df["Title"], _ = pd.factorize(df["Title"])

df['Embarked'] = df['Embarked'].map({'S': 0, 'C': 1, 'Q': 2})
X = df[['Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Fare', 'Embarked', 'FamilySize', 'Title']]

y = df['Survived']



X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=0)






param_grid = {
    'decisiontreeclassifier__max_depth': [3, 5, 7, 10, None],                # Limit tree depth; None means no limit
    'decisiontreeclassifier__min_samples_split': [2, 5, 10, 20],             # Minimum samples to split a node
    'decisiontreeclassifier__min_samples_leaf': [1, 2, 4, 8],                 # Minimum samples at a leaf node
    'decisiontreeclassifier__max_features': [None, 'sqrt', 'log2'],  # Number of features to consider at each split
    'decisiontreeclassifier__criterion': ['gini', 'entropy']                  # Splitting criteria
}

pipeline = make_pipeline(DecisionTreeClassifier(random_state=0, criterion='gini', max_depth=3, max_features=None, min_samples_leaf=4, min_samples_split=2))


grid = GridSearchCV(pipeline, param_grid, cv=5)  # 5-fold cross-validation
grid.fit(X_train, y_train)

print("Best Parameters:", grid.best_params_)
print("Best CV Score:", grid.best_score_)


pipeline.fit(X_train, y_train)

accuracy = pipeline.score(X_test, y_test)
print(f'Accuracy: {accuracy:.4f}')

y_pred = pipeline.predict(X_test)

#Visualize

'''fig, axs = plt.subplots(1, 2, figsize=(12, 5))
title_survival.plot(kind="bar", color="skyblue", ax=axs[0])
axs[0].set_title("Average Survival Rate by Title")
axs[0].set_ylabel("Survival Rate")
axs[0].tick_params(axis='x', rotation=45)



bars = axs[1].bar(title_counts.index, title_counts.values, color='skyblue')

# Annotate the bars with counts
for bar in bars:
    height = bar.get_height()
    axs[1].annotate(f'{int(height)}',
                 xy=(bar.get_x() + bar.get_width() / 2, height),
                 xytext=(0, 5),  # distance from top of bar
                 textcoords="offset points",
                 ha='center', va='bottom', fontsize=9)

# Labels
axs[1].set_title('Passenger Counts per Title')
axs[1].set_xlabel('Title')
axs[1].set_ylabel('Count')
axs[1].tick_params(axis='x', rotation=45)

plt.tight_layout()
plt.savefig('Title-survival.png')
plt.show()


sklearn.tree.plot_tree(dt, filled = True, feature_names=X.columns, class_names=["Dead", "Survived"])
plt.savefig("titanic-tree.png", dpi=300)
plt.show()'''






train_accuracy = pipeline.score(X_train, y_train)
test_accuracy = pipeline.score(X_test, y_test)

print(f"Train Accuracy: {train_accuracy:.4f}")
print(f"Test Accuracy: {test_accuracy:.4f}")

print(classification_report(y_test, y_pred))
print(confusion_matrix(y_test,y_pred))