import numpy as np
import matplotlib.pyplot as plt
from sklearn import decomposition
from sklearn import datasets

np.random.seed(5)

iris = datasets.load_diabetes()
X = iris.data
y = iris.target

pca = decomposition.PCA(n_components=10)
pca.fit(X)
X = pca.transform(X)

# Get the explained variance ratios for each PCA component
variance_ratios = pca.explained_variance_ratio_

# Plot the variance percentages in a barchart
plt.bar(range(len(variance_ratios)), variance_ratios)
plt.xlabel('PCA component')
plt.ylabel('Variance percentage')
plt.title('Variance percentages of PCA components')
plt.show()