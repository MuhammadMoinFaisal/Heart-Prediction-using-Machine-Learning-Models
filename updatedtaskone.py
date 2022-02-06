# To Perform Exploratory Data Analysis on the Data and to Predict the Output
# I will follow the following various steps given as 
# 1- Collecting Data        
# 2- Analyzing Data
# 3- Data Wrangling
# 4- Train & Test
# 5- Predict the output


import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.model_selection import RandomizedSearchCV, GridSearchCV

#Load the CSV File
data = pd.read_csv(r"C:\UpworkProjects\DwayneSalo\heart.csv")

print(data)


print(data.columns)

print(data.dtypes)

print(data.info())

print(data.isnull().sum())

# Analyzing Data 

print(data)

sns.countplot(x = 'sex', data = data)

sns.countplot(x = 'cp', hue = 'sex', data = data)

sns.catplot(x = 'cp', hue = 'sex', col = "fbs", data = data, kind = "count")

sns.countplot(x = 'exang', data = data)

sns.countplot(x = 'sex', hue = 'exang', data = data)

# Data Wrangling

print(data.isnull().sum())

sns.heatmap(data.isnull(), yticklabels = False, cmap = "viridis")

# Applying Machine Learning Algorithm

X  = data.drop("target", axis = 'columns')
print(X)
y = data["target"]
print(y)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 1)

#K Neighbors Classifier

from sklearn.neighbors import KNeighborsClassifier
knn = KNeighborsClassifier(n_neighbors = 24, metric = 'euclidean')

knn.fit(X_train, y_train)

y_pred = knn.predict(X_test)

from sklearn.metrics import classification_report, confusion_matrix, f1_score, accuracy_score

print(classification_report(y_test, y_pred))

print(confusion_matrix(y_test, y_pred))
print(accuracy_score(y_test, y_pred))


#Comparing Error Rate with the K Value


error = []

# Calculating error for K values between 1 and 40
for i in range(1, 40):
    knn = KNeighborsClassifier(n_neighbors=i)
    knn.fit(X_train, y_train)
    pred_i = knn.predict(X_test)
    error.append(np.mean(pred_i != y_test))

plt.figure(figsize=(12, 6))
plt.plot(range(1, 40), error, color='red', linestyle='dashed', marker='o',
         markerfacecolor='blue', markersize=10)
plt.title('Error Rate K Value')
plt.xlabel('K Value')
plt.ylabel('Mean Error')

#Decision Tree Classifier

from sklearn.tree import DecisionTreeClassifier
decision_tree = DecisionTreeClassifier()
decision_tree.fit(X_train, y_train)
Y_pred = decision_tree.predict(X_test)
acc_decision_tree1 = round(decision_tree.score(X_test, y_test) * 100, 2)
sk_report = classification_report(
    digits=6,
    y_true=y_test, 
    y_pred=Y_pred)
print("Accuracy", acc_decision_tree1)
print(sk_report)
### Confusion Matrix 
pd.crosstab(y_test, Y_pred, rownames=['Actual'], colnames=['Predicted'], margins=True)

from sklearn.ensemble import RandomForestClassifier
#Random Forest Classifier
rf_grid = {"n_estimators": np.arange(10, 1000, 50),
           "max_depth": [None, 3, 5, 10],
           "min_samples_split": np.arange(2, 20, 2),
           "min_samples_leaf": np.arange(1, 20, 2)}

# random hyperparameter search for RandomForestClassifier
rs_rf = RandomizedSearchCV(RandomForestClassifier(),
                           param_distributions=rf_grid,
                           cv=5,
                           n_iter=20,
                           verbose=True)

rs_rf.fit(X_train, y_train);
# the best parameters for RandomForestClassifier
print(rs_rf.best_params_)

rf_grid = {'n_estimators': 810, 'min_samples_split': 18, 'min_samples_leaf': 3, 'max_depth': 3}

# random hyperparameter search for RandomForestClassifier
rs_rf = RandomizedSearchCV(RandomForestClassifier(),
                           param_distributions=rf_grid,
                           cv=5,
                           n_iter=20,
                           verbose=True)

rs_rf.fit(X_train, y_train);

# Evaluating the randomized search random forest model
print(rs_rf.score(X_test, y_test))
