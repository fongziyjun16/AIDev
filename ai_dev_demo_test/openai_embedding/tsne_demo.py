import pandas as pd
import ast
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
from sklearn.manifold import TSNE

data_file = "data/AmazonFineFoodReviews_1k_embeddings.csv"

df = pd.read_csv(data_file, index_col=0)
# data type of embedding is string, convert to vector
df["embedding_vec"] = df["embedding"].apply(ast.literal_eval)
assert df["embedding_vec"].apply(len).nunique() == 1
matrix = np.vstack(df["embedding_vec"].values)

tsne = TSNE(n_components=2, perplexity=15, random_state=42, init='random', learning_rate=200)
vis_dims = tsne.fit_transform(matrix)

colors = ["red", "darkorange", "gold", "turquoise", "darkgreen"]
x = [x for x,_ in vis_dims]
y = [y for _,y in vis_dims]
colors_indices = df.Score.values - 1

assert len(vis_dims) == len(df.Score.values)

colormap = matplotlib.colors.ListedColormap(colors)
plt.scatter(x, y, c=colors_indices, cmap=colormap, alpha=0.3)
plt.title("Amazon ratings visualized in language using t-SNE")

plt.show()