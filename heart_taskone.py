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

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 3)

import tensorflow as tf
from tensorflow import keras
from sklearn.metrics import classification_report, confusion_matrix, f1_score, accuracy_score

#Now let us create a neural network
model = keras.Sequential([
    keras.layers.Dense(60, input_shape = (13,), activation = 'relu'),
    keras.layers.Dense(50, activation = 'relu'),
    keras.layers.Dense(50, activation = 'relu'),
    keras.layers.Dense(1, activation = 'sigmoid'),
    
    
    ])

model.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])

model.fit(X_train, y_train, epochs = 100)
model.evaluate(X_test, y_test)
yp = model.predict(X_test)
print(yp[:5])
y_pred = []
for element in yp:
    if element > 0.5:
        y_pred.append(1)
    else:
        y_pred.append(0)
y_pred[:10]
print("ANN Prediction",y_pred)

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

#rf_grid = {'n_estimators'= 810, 'min_samples_split'= 18, 'min_samples_leaf'= 3, 'max_depth'= 3}

# random hyperparameter search for RandomForestClassifier
rs_rf = RandomForestClassifier(n_estimators = 810, min_samples_split= 18, min_samples_leaf= 3, max_depth= 3)

rs_rf.fit(X_train, y_train);

# Evaluating the randomized search random forest model
print(rs_rf.score(X_test, y_test))

from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
model_scores = {}
pipe = make_pipeline(StandardScaler(), RandomForestClassifier(n_estimators = 810, min_samples_split= 18, min_samples_leaf= 3, max_depth= 3))

pipe.fit(X_train, y_train)      

model_scores[' RandomForestClassifier'] = pipe.score(X_test, y_test)  

print(model_scores)

from sklearn.linear_model import LogisticRegression

#  LogisticRegression hyperparameters

log_reg_grid = {"C": np.logspace(-4, 4, 20),
                "solver": ["liblinear"]}

#   hyperparameter tuning  for LogisticRegression
rs_log_reg = RandomizedSearchCV(LogisticRegression(),
                                param_distributions=log_reg_grid,
                                cv=5,
                                n_iter=20,
                                verbose=True)

rs_log_reg.fit(X_train, y_train);
print(rs_log_reg.best_params_)
print("Logistic Regression",rs_log_reg.score(X_test, y_test))

from sklearn.naive_bayes import GaussianNB

#  Gaussian Naive Bayes hyperparameters
nb_grid = {
    'var_smoothing': np.logspace(0,-9, num=100)
}

# Hyper Parameter  Tuning  for Gausian NB
rs_gnb = RandomizedSearchCV(GaussianNB(),
                           param_distributions=nb_grid,
                           cv=5,
                           n_iter=20,
                           verbose=True)

rs_gnb.fit(X_train, y_train);

print(rs_gnb.best_params_)
model = GaussianNB(var_smoothing =  5.3366992312063123e-05)
model.fit(X_train, y_train);
print("Gaussian Naive Bayes",model.score(X_test, y_test))

