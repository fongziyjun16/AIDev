import jieba
import numpy as np
import matplotlib.pyplot as plt

corpus = [
    "我特别特别喜欢看电影",
    "这部电影真的是很好看的电影",
    "今天天气真好是难得的好天气",
    "我今天去看了一部电影",
    "电影院的电影都很好看"
]

corpus_tokenized = [list(jieba.cut(sentence)) for sentence in corpus]

word_dict = {}

for sentence in corpus_tokenized:
    for word in sentence:
        if word not in word_dict:
            word_dict[word] = len(word_dict)

bow_vectors = []

for sentence in corpus_tokenized:
    sentence_vector = [0] * len(word_dict)
    for word in sentence:
        sentence_vector[word_dict[word]] += 1
    bow_vectors.append(sentence_vector)

def cosine_similarity(vec1, vec2):
    dot_product = np.dot(vec1, vec2)
    norm_a = np.linalg.norm(vec1)
    norm_b = np.linalg.norm(vec2)
    return dot_product / (norm_a * norm_b)

similarity_matrix = np.zeros((len(corpus), len(corpus)))
for i in range(len(corpus)):
    for j in range(len(corpus)):
        similarity_matrix[i][j] = cosine_similarity(bow_vectors[i], bow_vectors[j])

plt.rcParams["font.family"] = ['Songti SC']
plt.rcParams["font.sans-serif"] = ['Songti SC']
plt.rcParams["axes.unicode_minus"] = False
fig, ax = plt.subplots()
cax = ax.matshow(similarity_matrix, cmap=plt.cm.Blues)
fig.colorbar(cax)
ax.set_xticks(range(len(corpus)))
ax.set_yticks(range(len(corpus)))
ax.set_xticklabels(corpus, rotation=45, ha="left")
ax.set_yticklabels(corpus)
plt.show()

