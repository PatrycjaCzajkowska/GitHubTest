from sklearn.datasets import load_boston, load_diabetes
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import cross_val_score

#zad. 1
# Loading a set of property features and their prices
propertyOfBoston = load_boston()
# conversion to pandas.DataFrame
listPropertyOfBoston = pd.DataFrame(propertyOfBoston['data'], columns=propertyOfBoston['feature_names'])
print('listPropertyOfBoston:')
print(listPropertyOfBoston)
# appending price information to the rest of the dataframe
listPropertyOfBoston['target'] = np.array(list(propertyOfBoston['target']))

# printing data
print('')
print('Example values of features:')
print(propertyOfBoston.data[:3])
print('')
print('Example values:')
print(propertyOfBoston.target[:3])
print('')
print('Elements of the set:')
print(list(propertyOfBoston.keys()))
print('')
print('Keys in the data set:')
print(propertyOfBoston.keys())
print('')
print('propertyOfBoston.DESCR')
print(propertyOfBoston.DESCR)

#No. of rooms
rooms = propertyOfBoston['data'][:, np.newaxis, 3]
plt.scatter(rooms, propertyOfBoston['target'])
plt.show()

# Creation of a linear regressor
linreg = LinearRegression()
X_train, X_test, y_train, y_test = train_test_split(rooms, propertyOfBoston['target'], test_size = 0.3)
linreg.fit(X_train, y_train)

# price prediction
y_pred = linreg.predict(X_test)

# default metric
print('Default metric: ', linreg.score(X_test, y_test))

# indicator (metric) r^2
print('Metric r2: ', r2_score(y_test, y_pred))

# regression coefficients
print('Regression coefficients', linreg.coef_)

# regression graph
plt.scatter(X_test, y_test, color='black')
plt.plot(X_test, y_pred, color='blue', linewidth=3)
plt.show()

# metric 1
cv_score_r2 = cross_val_score(linreg, rooms, propertyOfBoston.target, cv=5, scoring='r2')
print('metric 1')
print(cv_score_r2)
print('')

# metric 2
cv_score_ev = cross_val_score(linreg, rooms, propertyOfBoston.target, cv=5, scoring='explained_variance')
print('metric 2')
print(cv_score_ev)
print('')

# metric 3
cv_score_mse = cross_val_score(linreg, rooms, propertyOfBoston.target, cv=5, scoring='neg_mean_squared_error')
print('metric 3')
print(cv_score_mse)
print('')

# metric 4
max_error = cross_val_score(linreg, rooms, propertyOfBoston.target, cv=5, scoring='neg_mean_squared_error')
print('metric 4')
print(max_error)

#zad. bonus

diabetics = load_diabetes()
listOfDiabetics = pd.DataFrame(diabetics['data'], columns=diabetics['feature_names'])
listOfDiabetics['target'] = np.array(list(diabetics['target']))

print('Elements of the set:')
print(list(diabetics.keys()))
print('Keys in the data set:')
print(diabetics.keys())
print('diabetics.DESCR')
print(diabetics.DESCR)


#age of diabetics
age = diabetics['data'][:, np.newaxis, 0]
plt.scatter(age, diabetics['target'])
plt.show()

# creation of a linear regressor
linreg = LinearRegression()
X_train, X_test, y_train, y_test = train_test_split(age, diabetics['target'], test_size = 0.5)
linreg.fit(X_train, y_train)

# predicting diabetes by age
y_pred = linreg.predict(X_test)

# default metric
print('default metric', linreg.score(X_test, y_test))

# indicator (metric) r ^ 2
print('metric r2: ', r2_score(y_test, y_pred))

# regression coefficients
print('regression coefficients', linreg.coef_)

# regression graph
plt.scatter(X_test, y_test, color='black')
plt.plot(X_test, y_pred, color='blue', linewidth=2)
plt.show()

# metric 1
cv_score_r2 = cross_val_score(linreg, age, diabetics.target, cv=5, scoring='r2')
print('cv_score_r2')
print(cv_score_r2)
print('')

# metric 2
cv_score_ev = cross_val_score(linreg, age, diabetics.target, cv=5, scoring='explained_variance')
print('cv_score_ev')
print(cv_score_ev)
print('')

# metric 3
cv_score_mse = cross_val_score(linreg, age, diabetics.target, cv=5, scoring='neg_mean_squared_error')
print('cv_score_mse')
print(cv_score_mse)
print()

# metric 4
max_error = cross_val_score(linreg, age, diabetics.target, cv=5, scoring='max_error')
print('max_error')
print(max_error)
print('')