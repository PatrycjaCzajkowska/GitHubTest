
from sklearn.datasets import fetch_openml
from sklearn.cluster import KMeans
from sklearn.model_selection import train_test_split
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt

# Zad. 1

cars = fetch_openml('cars1')

# visualization of all features
print(cars.keys())
print(cars['target'])
print(cars['DESCR'])
print(cars['feature_names'])
print(cars['url'])
print(cars['data'][0])
print(cars['categories'])
print(cars['details'])

# I divide the collection into features and labels
# I choose the features 1 and 5
X = cars.data[:, [0, 4]]
print(X)
y = cars['target']
y = [int(elem) for elem in y]
print(y)

# I use the function to divide the set into a training set and a test set
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.3)

# I create a classifier with four clusters (classes)
kmn = KMeans(n_clusters=4)

# I teach classifier on training data
kmn.fit(X_train)

# I extract cluster focal points
# I will show them on the graph next to the points from the training set
central = kmn.cluster_centers_
fig, ax = plt.subplots(1, 2)

# the first chart is our learning set, with real classes
ax[0].scatter(X_train[:, 0], X_train[:, 1], c=y_train, s=20)

# Now I use training data to check what the classifier thinks about them
y_pred_train = kmn.predict(X_train)
ax[1].scatter(X_train[:, 0], X_train[:, 1], c=y_pred_train, s=20)

# I add cluster centers on the second chart
ax[1].scatter(central[:, 0], central[:, 1], c='red', s=50)
plt.show()

# I try to predict car classes for the test set
y_pred = kmn.predict(X_test)

# New car classes provided by clustering
plt.scatter(X_test[:, 0], X_test[:, 1], c=y_pred, s=20)

# As above, I display cluster centers
plt.scatter(central[:, 0], central[:, 1], c='red', s=50)
plt.show()

#  Zad. 2

# W klastrach przedtawilam samoochody w roznych kombinacjach (moc/spalanie silnika i zasieg).
# Klaster najwyzszy to samochody o duzej mocy i krotkim zasiegu.
# Klaster najnizszy to samochody o malej mocy i dalekim zasiegu.
