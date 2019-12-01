

from sklearn import datasets
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import seaborn as sns

#Zadadnie 1
iris = datasets.load_iris()
print('Displaying elements of list: ', list(iris.keys()))
print('-----')
print('DESCR: contains information about the description of the file from which the data is collected', iris['DESCR'])
print('Typ of first element from \'DESCR\': ', type(iris['DESCR'][0]))
print('Five elements of DESCR', iris['DESCR'][0:5])
print('-----')
print('targer: is an array', iris['target'])
print('Typ of first element from \'target\': ', type(iris['target'][0]))
print('Five elements of target', iris['target'][0:5])
print('-----')
print('target_names: is an array which display types of iris - setosa, versicolor, virginica', iris['target_names'])
print('Typ of first element from \'target_names\': ', type(iris['target_names'][0]))
print('Three elements of target_names', iris['target_names'][0:3])
print('-----')
print('feature_names: displays information abouth length and width of sepal and length and width of petal in cm', iris['feature_names'])
print('Typ of first element from \'feature_names\': ', type(iris['feature_names'][0]))
print('Four elements of feature_names', iris['feature_names'][0:4])
print('-----')
print('data: is a array', iris['data'])
print('Typ of first element from \'data\': ', type(iris['data'][0]))
print('Three elements of data', iris['data'][0:3])
print('-----')
print('filename: displays information about the file from which the iris is taken and its location on the disk', iris['filename'])
print('Typ of first element from \'filename\': ', type(iris['filename'][0]))
print('Fifth elements of filename', iris['filename'][5])
print('-----')

# displaying of data set
iris_list = pd.DataFrame(iris['data'], columns=iris['feature_names'])
targets = map(lambda x: iris['target_names'][x], iris['target'])
iris_list['species'] = np.array(list(targets))
sns.pairplot(iris_list, hue='species')
plt.show()
iris_list.head(3)

#Zadadnie 2

# I Divide the collection into features and labels
X = iris.data
y = iris.target

# I use the function to divide the set into a training set and a test set
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.5)

# I create the k-NN classifier using the parameter 8 neighbors
knn = KNeighborsClassifier(n_neighbors = 8)

# I learn classifier on learning set
knn.fit(X_train, y_train)

# I predict values for the test set
y_pred = knn.predict(X_test)

print('I check the first few values provided')
print(y_pred[:8])

print('I check accuracy of the classifier')
print(knn.score(X_test, y_test))

# I create the plane of all possible values for features 0 and 2, in steps of 0.1
x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
y_min, y_max = X[:, 2].min() - 1, X[:, 2].max() + 1
xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.1),
                     np.arange(y_min, y_max, 0.1))

# I teach classifier on only two selected features
knn.fit(X_train[:, [0, 2]], y_train)

# I anticipate every point on the plane
Z = knn.predict(np.c_[xx.ravel(), yy.ravel()])
Z = Z.reshape(xx.shape)

print('I create a contourplot')
plt.contourf(xx, yy, Z, alpha=0.8, cmap=plt.cm.bwr)
plt.scatter(X[:, 0], X[:, 2], c=y, s=20, edgecolor='k')
plt.show()

# I create a list
list_n = [1,2,3,4,5,6,7,8]
precisions = []
for n_neighbors in list_n:

    knn = KNeighborsClassifier(n_neighbors)
    knn.fit(X_train, y_train)
    y_pred = knn.predict(X_test)
    precision = knn.score(X_test, y_test)
    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 2].min() - 1, X[:, 2].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.1),
                         np.arange(y_min, y_max, 0.1))
    precisions.append(precision)
    plt.contourf(xx, yy, Z, alpha=0.8, cmap=plt.cm.bwr)
    plt.scatter(X[:, 0], X[:, 2], c=y, s=20, edgecolor='k')
    plt.show()

print(precisions)
plt.plot(list_n, precisions)
plt.show()