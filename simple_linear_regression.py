# Simple Linear Regression
# regression means finding the relation btw two variables or more.
# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing the dataset
dataset = pd.read_csv('Salary_Data.csv')
X = dataset.iloc[:, :-1].values # x is years of experience (independent variable.)
y = dataset.iloc[:, 1].values # y is salary depend on (dependent variable)

# Splitting the dataset into the Training set and Test set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 1/3, random_state = 0)

# Feature Scaling
"""from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
X_train = sc_X.fit_transform(X_train)
X_test = sc_X.transform(X_test)
sc_y = StandardScaler()
y_train = sc_y.fit_transform(y_train)"""

# fitting simple linear regression to the Training set.
# Here we are creating the machine (simple linear regression machine ) which makes the relation 
# between years and salary and can predict salary according to years given.
from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
# training data(independent variable) is X_train , target_values(dependent variable) is y_train.
regressor.fit(X_train,y_train)

# predictiong the test set results
y_pred = regressor.predict(X_test) 

# visualizing the training set result.
# scatter function will draw the observation point in graph.
plt.scatter(X_train,y_train,color='red')    
# to plot regression line we have plt.plot()
plt.plot(X_train,regressor.predict(X_train),color='blue')
plt.title('Salary Vs Experience (Training Set)')
plt.xlabel('Years of Experience')
plt.ylabel('Salary')
plt.show()
# note:- red points shown in the graph are the real values and projection of those  red points on the line is the predicted values by our model.. 