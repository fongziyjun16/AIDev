import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt

sentences = [
    "Kage is Teacher",
    "Mazong is Boss",
    "Niuzong is Boss",
    "Xiaobing is Student",
    "Xiaoxue is Student",
]

words = ' '.join(sentences).split()
word_list = list(set(words))
word_to_idx = {word: idx for idx, word in enumerate(word_list)}
idx_to_word = {idx: word for idx, word in enumerate(word_list)}
voc_size = len(word_list)

def create_cow_dataset(sentences, windows_size=2):
    data = []
    for sentence in sentences:
        sentence = sentence.split()
        for idx, word in enumerate(sentence):
            context_word = \
                sentence[max(idx - windows_size, 0):idx] + \
                sentence[idx+1:min(idx + windows_size + 1, len(sentence))]
            data.append((word, context_word))
    return data

cbow_data = create_cow_dataset(sentences)
# print(cbow_data)

def one_hot_encoding(word, word_to_idx):
    tensor = torch.zeros(len(word_to_idx)) # 创建一个长度与词汇表相同的全 0 张量
    tensor[word_to_idx[word]] = 1  # 将对应词的索引设为 1
    return tensor  # 返回生成的 One-Hot 向量

class CBOW(nn.Module):
    def __init__(self, voc_size, embedding_size):
        super(CBOW, self).__init__()
        self.input_to_hidden = nn.Linear(voc_size, embedding_size, bias=False)
        self.hidden_to_output = nn.Linear(embedding_size, voc_size, bias=False)

    def forward(self, X):
        embedding = self.input_to_hidden(X)
        hidden_layer = torch.mean(embedding, dim=0)
        output_layer = self.hidden_to_output(hidden_layer.unsqueeze(0))
        return output_layer

embedding_size = 2
cbow_model = CBOW(voc_size, embedding_size)

# for target, context_words in cbow_data:
#     X = torch.stack([one_hot_encoding(word, word_to_idx) for word in context_words]).float()
#     y_true = torch.tensor([word_to_idx[target]], dtype=torch.long)

learning_rate = 0.001
epochs = 1000
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(cbow_model.parameters(), lr=learning_rate)
loss_values = []
for epoch in range(epochs):
    loss_sum = 0
    for target, context_words in cbow_data:
        X = torch.stack([one_hot_encoding(word, word_to_idx) for word in context_words]).float()
        y_true = torch.tensor([word_to_idx[target]], dtype=torch.long)
        y_pred = cbow_model(X)
        loss = criterion(y_pred, y_true)
        loss_sum += loss.item()
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    if (epoch + 1) % 100 == 0:
        print(f"Epoch {epoch + 1}, Loss: {loss_sum / len(cbow_data)}")
        loss_values.append(loss_sum / len(cbow_data))

plt.rcParams["font.family"]=['SimHei'] # 用来设定字体样式
plt.rcParams['font.sans-serif']=['SimHei'] # 用来设定无衬线字体样式
plt.rcParams['axes.unicode_minus']=False # 用来正常显示负号
plt.plot(range(1, epochs//100 + 1), loss_values) # 绘图
plt.title(' 训练损失曲线 ') # 图题
plt.xlabel(' 轮次 ') # X 轴 Label
plt.ylabel(' 损失 ') # Y 轴 Label
plt.show() # 显示图