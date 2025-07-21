from sklearn.cluster import KMeans
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from sklearn.metrics import silhouette_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
import numpy as np
import pandas as pd


df = pd.read_csv("train.csv")


print(df.head())

# easier and more efficient to take care of this though a OneHotEncoder in the pipeline, easier if all transformers belong to the pipeline
'''df['Sex'], _ = pd.factorize(df['Sex'])
df['Embarked'], _ = pd.factorize(df['Embarked'])'''


X = df[['Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Fare', 'Embarked']]
y = df['Survived']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=0)

# inefficient, easier to do it with the pipeline

'''print(X_train.isnull().sum())'''

'''imputer = SimpleImputer(strategy='mean')
X_train_imputed = imputer.fit_transform(X_train)
X_test_imputed = imputer.transform(X_test)

scaler = StandardScaler()

X_train_imputed_scaled = scaler.fit_transform(X_train_imputed)
X_test_imputed_scaled = scaler.transform(X_test_imputed)


kmeans = KMeans(random_state=0, n_clusters=2, n_init='auto')
kmeans.fit(X_train_imputed_scaled)
y_pred = kmeans.predict(X_test_imputed_scaled)'''

numerical_cols = ['Pclass', 'Age', 'SibSp', 'Parch', 'Fare']
categorial_cols = ['Sex', 'Embarked']

# routing system to do diff transformations on diff cols, later fed into the main pipeline
preprocessor = ColumnTransformer(transformers = [
    ('num', Pipeline([
        ('imputer', SimpleImputer(strategy='mean')), 
        ('scaler', StandardScaler())]), 
        numerical_cols),
    ('cat', OneHotEncoder(handle_unknown = 'ignore'), categorial_cols)
])

pipeline = Pipeline([
    ('preprocessor', preprocessor),
    ('kmeans', KMeans(n_clusters=2, random_state=0, n_init=100))
])

pipeline.fit(X_train) # does the preprocessing on the data behind the scenes.

y_pred = pipeline.predict(X_test) # does the preprocessing on the data behind the scenes.


inertia = pipeline[-1].inertia_ # get it from the last step

X_test_transformed = pipeline[:-1].transform(X_test) # transform just runs though all the steps of pipeline but the [:-1] tells it to do all the steps but the last one

sil_score = silhouette_score(X_test_transformed, y_pred)

print(f"Inertia: {inertia:.2f}") # scale dependent, if i didn't scale this data this would mean nothing
print(f"Silhouette Coefficent Score: {sil_score:.4f}") # best interpreted against diff k values