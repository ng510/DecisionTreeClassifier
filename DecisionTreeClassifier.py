# Datenquelle: https://www.kaggle.com/uciml/mushroom-classification

# Klassifizierung mittels eines Entscheidungsbaumes, ob Pilz essbar oder nicht.

import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.tree import export_graphviz

import graphviz

df = pd.read_csv("./mushrooms.csv")
df = pd.get_dummies(df)
df = df.drop("class_e", axis = 1)

print(df.head())

X = df.drop("class_p", axis = 1).values
y = df["class_p"].values

# Train-Test
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state = 0, test_size = 0.25)

# Entscheidungsbaum
model = DecisionTreeClassifier(criterion = "entropy")
model.fit(X_train, y_train)

print(model.score(X_test, y_test))

# Grafische Ausgabe des Baumes
tree = export_graphviz(model, None, 
                       feature_names = df.drop("class_p", axis = 1).columns.values,
                       class_names = ["essbar", "nicht essbar"],
                       rounded = True,
                       filled = True)
 
graphviz.Source(tree)