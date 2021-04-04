import numpy as np
import seaborn as sb
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.datasets.samples_generator import make_blobs

x,y_true = make_blobs(n_samples = 800, centers=5, cluster_std = 0.45, random_state=0)
plt.scatter(x[:,0],x[:,1], s=30)

func = KMeans(n_clusters=5)

func.fit(x)
y_func=func.predict(x)

plt.scatter(x[:,0],x[:,1], c=y_func ,s=15)
center=func.cluster_centers_
plt.scatter(center[:,0],center[:,1], c='red', s=75, alpha = 1)
plt.show()
