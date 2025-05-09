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

def create_skipgram_dataset(sentences, window_size=2):
    data = []
    for sentence in sentences:
        sentence = sentence.split()
        for idx, word in enumerate(sentence):
            for neighbor in sentence[max(0, idx - window_size):min(idx + window_size + 1, len(sentence))]:
                if neighbor != word:
                    data.append((word, neighbor))
    return data
skipgram_data = create_skipgram_dataset(sentences)

def one_hot_encoding(word, word_to_idx):
    tensor = torch.zeros(len(word_to_idx))
    tensor[word_to_idx[word]] = 1
    return tensor

class SkipGram(nn.Module):
    def __init__(self, voc_size, embedding_size):
        super(SkipGram, self).__init__()
        self.input_to_hidden = nn.Linear(voc_size, embedding_size, bias=False)
        self.hidden_to_output = nn.Linear(embedding_size, voc_size, bias=False)

    def forward(self, X): # X like (batch_size, voc_size)
        hidden = self.input_to_hidden(X)
        output = self.hidden_to_output(hidden)
        return output

embedding_size = 2
skipgram_model = SkipGram(voc_size, embedding_size)
# print(skipgram_model)

learning_rate = 0.001
epochs = 1000
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(skipgram_model.parameters(), lr=learning_rate)
loss_values = []
for epoch in range(epochs):
    loss_sum = 0
    for context, target in skipgram_data:
        X = one_hot_encoding(target, word_to_idx).float().unsqueeze(0)
        y_true = torch.tensor([word_to_idx[context]], dtype=torch.long)
        y_pred = skipgram_model(X)
        loss = criterion(y_pred, y_true)
        loss_sum += loss.item()
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    if (epoch + 1) % 100 == 0:
        print(f"Epoch {epoch + 1}, Loss: {loss_sum / len(skipgram_data)}")
        loss_values.append(loss_sum / len(skipgram_data))

plt.rcParams["font.family"] = ["Songti SC"]
plt.rcParams["font.sans-serif"] = ["Songti SC"]
plt.rcParams["axes.unicode_minus"] = False
plt.plot(range(1, epochs//100 + 1), loss_values)
plt.title("训练损失曲线")
plt.xlabel("轮次")
plt.ylabel("损失")
plt.show()

fig, ax = plt.subplots()
for word, idx in word_to_idx.items():
    vec = skipgram_model.input_to_hidden.weight[:,idx].detach().numpy()
    ax.scatter(vec[0], vec[1])
    ax.annotate(word, (vec[0], vec[1]), fontsize=12)
plt.title("二维词嵌入")
plt.xlabel("向量维度 1")
plt.ylabel("向量维度 2")
plt.show()
