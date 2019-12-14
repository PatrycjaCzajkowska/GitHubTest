
# Zad.1:
# Uwazam, ze ksztalt platka - jego dlugosc i szerokosc jest cecha,
# ktora zostala wywnioskowana przez PCA.
# Cecha ta moze oznaczac powierzchnie platka.

# Zad.bonus:
from sklearn.datasets import fetch_openml
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from sklearn.decomposition import PCA

cars = fetch_openml('cars1')
# I divide the collection into features and labels
# I leave all the features - I will try to guess which features are the most important
X = cars.data
y = cars.target

# Initialization. I fill in n_components, I use all the features pca = PCA (n_components = 3)
pca = PCA()
pca.fit(X)

# Analysis (decomposition) of PCA creates n new "artificial" features that try their best reflect
# the variability of the original set
print("Number of components:")
print( pca.n_components_)

# Additionally, I can check what impact our original features have on inferred new features
print("Composition of new features:")
print(pca.components_)

# Finally, I can determine which new, deduced traits have the greatest impact on the overall variability of the set
print(pca.explained_variance_ratio_)

# Charts will be created using the seaborn package
# import seaborn as sns

# Conversion to pandas.DataFrame
cars_df = pd.DataFrame(cars['data'], columns=cars['feature_names'])

# Appending species information to the rest of the dataframe
cars_df['target'] = np.array(list(cars['target']))

# chart
# sns.pairplot(cars_df, hue='target')
plt.show()

# I will try to reduce our set of features to the only one, the best
pca_limit = PCA(n_components = 1)
X_new = pca_limit.fit_transform(X)
X_new[:6]

# Features:
print("Number of components: ")
print(pca_limit.n_components_)

# Impact of the original features to deduce the feature
print("Composition of the new feature:")
print(pca_limit.components_)

# "The explainability" of the new feature is still very high
print(pca_limit.explained_variance_ratio_)
plt.scatter(X_new, y)
plt.show()