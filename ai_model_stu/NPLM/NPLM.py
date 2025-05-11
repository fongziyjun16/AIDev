import random

import torch
import torch.nn as nn
import torch.optim as optim

sentences = ["我 喜欢 玩具", "我 爱 爸爸", "我 讨厌 挨打"]
word_list = list(set(" ".join(sentences).split()))
word_to_idx = {word: idx for idx, word in enumerate(word_list)}
idx_to_word = {idx: word for idx, word in enumerate(word_list)}
voc_size = len(word_list)

batch_size = 2
def make_batch():
    input_batch = []
    target_batch = []
    selected_sentences = random.sample(sentences, batch_size)
    for sen in selected_sentences:
        word = sen.split()
        input = [word_to_idx[n] for n in word[:-1]]
        target = word_to_idx[word[-1]]
        input_batch.append(input)
        target_batch.append(target)
    input_batch = torch.LongTensor(input_batch)
    target_batch = torch.LongTensor(target_batch)
    return input_batch, target_batch

input_batch, target_batch = make_batch()
input_words = []
for input_idx in input_batch:
    input_words.append([idx_to_word[idx.item()] for idx in input_idx])
target_words = [idx_to_word[idx.item()] for idx in target_batch]

class NPLM(nn.Module):
    def __init__(self, voc_size, embedding_size, n_step, n_hidden):
        super(NPLM, self).__init__()
        self.voc_size = voc_size
        self.embedding_size = embedding_size
        self.n_step = n_step
        self.n_hidden = n_hidden

        self.C = nn.Embedding(voc_size, embedding_size)
        self.linear1 = nn.Linear(n_step * embedding_size, n_hidden)
        self.linear2 = nn.Linear(n_hidden, voc_size)
    def forward(self, X):
        X = self.C(X)
        X = X.view(-1, self.n_step * self.embedding_size)
        hidden = torch.tanh(self.linear1(X))
        output = self.linear2(hidden)
        return output

n_step = 2 # 时间步数，表示每个输入序列的长度，也就是上下文长度
n_hidden = 2 # 隐藏层大小
embedding_size = 2 # 词嵌入大小
model = NPLM(voc_size, embedding_size, n_step, n_hidden) # 创建神经概率语言模型实例

criterion = nn.CrossEntropyLoss() # 定义损失函数为交叉熵损失
optimizer = optim.Adam(model.parameters(), lr=0.1) # 定义优化器为 Adam，学习率为 0.1
# 训练模型
for epoch in range(5000): # 设置训练迭代次数
   optimizer.zero_grad() # 清除优化器的梯度
   input_batch, target_batch = make_batch() # 创建输入和目标批处理数据
   output = model(input_batch) # 将输入数据传入模型，得到输出结果
   loss = criterion(output, target_batch) # 计算损失值
   if (epoch + 1) % 1000 == 0: # 每 1000 次迭代，打印损失值
     print('Epoch:', '%04d' % (epoch + 1), 'cost =', '{:.6f}'.format(loss))
   loss.backward() # 反向传播计算梯度
   optimizer.step() # 更新模型参数

# 进行预测
input_strs = [['我', '讨厌'], ['我', '喜欢']]  # 需要预测的输入序列
# 将输入序列转换为对应的索引
input_indices = [[word_to_idx[word] for word in seq] for seq in input_strs]
# 将输入序列的索引转换为张量
input_batch = torch.LongTensor(input_indices)
# 对输入序列进行预测，取输出中概率最大的类别
predict = model(input_batch).data.max(1)[1]
# 将预测结果的索引转换为对应的词
predict_strs = [idx_to_word[n.item()] for n in predict.squeeze()]
for input_seq, pred in zip(input_strs, predict_strs):
   print(input_seq, '->', pred)  # 打印输入序列和预测结果

