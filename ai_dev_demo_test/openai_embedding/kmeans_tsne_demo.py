import pandas as pd
import ast
import numpy as np
from sklearn.cluster import KMeans
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import matplotlib

data_file = "data/AmazonFineFoodReviews_1k_embeddings.csv"

df = pd.read_csv(data_file, index_col=0)
# data type of embedding is string, convert to vector
df["embedding_vec"] = df["embedding"].apply(ast.literal_eval)
assert df["embedding_vec"].apply(len).nunique() == 1
matrix = np.vstack(df["embedding_vec"].values)

n_clusters = 4
# n_clusters = 3
# n_clusters = 5
kmeans = KMeans(n_clusters = n_clusters, init='k-means++', random_state=42, n_init=10)
kmeans.fit(matrix)

df['Cluster'] = kmeans.labels_

colors = ["red", "green", "blue", "purple"]
# colors = ["red", "green", "blue"]
# colors = ["red", "green", "blue", "purple", "orange"]

tsne = TSNE(n_components=2, random_state=42)
vis_data = tsne.fit_transform(matrix)

x = vis_data[:, 0]
y = vis_data[:, 1]

color_indices = df['Cluster'].values
colormap = matplotlib.colors.ListedColormap(colors)
plt.scatter(x, y, c=color_indices, cmap=colormap)
plt.title("Clustering visualized in 2D using t-SNE")
plt.show()