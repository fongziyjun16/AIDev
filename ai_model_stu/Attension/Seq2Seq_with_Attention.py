sentences = [
    ['咖哥 喜欢 小冰', '<sos> KaGe likes XiaoBing', 'KaGe likes XiaoBing <eos>'],
    ['我 爱 学习 人工智能', '<sos> I love studying AI', 'I love studying AI <eos>'],
    ['深度学习 改变 世界', '<sos> DL changed the world', 'DL changed the world <eos>'],
    ['自然 语言 处理 很 强大', '<sos> NLP is so powerful', 'NLP is so powerful <eos>'],
    ['神经网络 非常 复杂', '<sos> Neural-Nets are complex', 'Neural-Nets are complex <eos>']]
word_list_cn, word_list_en = [], []  # 初始化中英文词汇表
for s in sentences:
    word_list_cn.extend(s[0].split())
    word_list_en.extend(s[1].split())
    word_list_en.extend(s[2].split())
word_list_cn = list(set(word_list_cn))
word_list_en = list(set(word_list_en))
word2idx_cn = {w: i for i, w in enumerate(word_list_cn)}
word2idx_en = {w: i for i, w in enumerate(word_list_en)}
idx2word_cn = {i: w for i, w in enumerate(word_list_cn)}
idx2word_en = {i: w for i, w in enumerate(word_list_en)}
voc_size_cn = len(word_list_cn)
voc_size_en = len(word_list_en)

import numpy as np
import torch
import random
def make_data(sentences):
    random_sentence = random.choice(sentences)
    encoder_input = np.array([[word2idx_cn[n] for n in random_sentence[0].split()]])
    decoder_input = np.array([[word2idx_en[n] for n in random_sentence[1].split()]])
    target = np.array([[word2idx_en[n] for n in random_sentence[2].split()]])
    encoder_input = torch.LongTensor(encoder_input)
    decoder_input = torch.LongTensor(decoder_input)
    target = torch.LongTensor(target)
    return encoder_input, decoder_input, target
encoder_input, decoder_input, target = make_data(sentences)
for s in sentences:
    if all([word2idx_cn[w] in encoder_input[0] for w in s[0].split()]):
        original_sentence = s
        break

import torch.nn as nn
class Attention(nn.Module):
    def __init__(self):
        super(Attention, self).__init__()

    def forward(self, decoder_context, encoder_context):
        scores = torch.matmul(decoder_context, encoder_context.transpose(-2, -1))
        attn_weights = nn.functional.softmax(scores, dim=-1)
        context = torch.matmul(attn_weights, encoder_context)
        return context, attn_weights

class Encoder(nn.Module):
    def __init__(self, input_size, hidden_size):
        super(Encoder, self).__init__()
        self.hidden_size = hidden_size
        self.embedding = nn.Embedding(input_size, hidden_size)
        self.rnn = nn.RNN(hidden_size, hidden_size, batch_first=True)
    def forward(self, inputs, hidden):
        embedded = self.embedding(inputs)
        output, hidden = self.rnn(embedded, hidden)
        return output, hidden
class DecoderWithAttention(nn.Module):
    def __init__(self, hidden_size, output_size):
        super(DecoderWithAttention, self).__init__()
        self.hidden_size = hidden_size
        self.embedding = nn.Embedding(output_size, hidden_size)
        self.rnn = nn.RNN(hidden_size, hidden_size, batch_first=True)
        self.attention = Attention()
        self.out = nn.Linear(2 * hidden_size, output_size)
    def forward(self, dec_inputs, hidden, enc_output):
        embedded = self.embedding(dec_inputs)
        rnn_output, hidden = self.rnn(embedded, hidden)
        context, attn_weights = self.attention(rnn_output, enc_output)
        dec_output = torch.cat((rnn_output, context), dim=-1)
        dec_output = self.out(dec_output)
        return dec_output, hidden, attn_weights

n_hidden = 128
encoder = Encoder(voc_size_cn, n_hidden)
decoder = DecoderWithAttention(n_hidden, voc_size_en)

class Seq2Seq(nn.Module):
    def __init__(self, encoder, decoder):
        super(Seq2Seq, self).__init__()
        self.encoder = encoder
        self.decoder = decoder
    def forward(self, enc_input, hidden, dec_input):
        encoder_output, encoder_hidden = self.encoder(enc_input, hidden)
        decoder_hidden = encoder_hidden
        decoder_output, _, attn_weights = self.decoder(dec_input, decoder_hidden, encoder_output)
        return decoder_output, attn_weights
model = Seq2Seq(encoder, decoder)

# 定义训练函数
def train_seq2seq(model, criterion, optimizer, epochs):
    for epoch in range(epochs):
       encoder_input, decoder_input, target = make_data(sentences) # 训练数据的创建
       hidden = torch.zeros(1, encoder_input.size(0), n_hidden) # 初始化隐藏状态
       optimizer.zero_grad()# 梯度清零
       output, _ = model(encoder_input, hidden, decoder_input) # 获取模型输出
       loss = criterion(output.view(-1, voc_size_en), target.view(-1)) # 计算损失
       if (epoch + 1) % 40 == 0: # 打印损失
          print(f"Epoch: {epoch + 1:04d} cost = {loss:.6f}")
       loss.backward()# 反向传播
       optimizer.step()# 更新参数

epochs = 400
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
train_seq2seq(model, criterion, optimizer, epochs)

import matplotlib.pyplot as plt # 导入 matplotlib
import seaborn as sns # 导入 seaborn
plt.rcParams["font.family"]=['SimHei'] # 用来设定字体样式
plt.rcParams['font.sans-serif']=['SimHei'] # 用来设定无衬线字体样式
plt.rcParams['axes.unicode_minus']=False #  用 来 正 常 显 示 负 号
def  visualize_attention(source_sentence, predicted_sentence, attn_weights):
    plt.figure(figsize=(10, 10)) # 画布
    ax = sns.heatmap(attn_weights, annot=True, cbar=False,
                     xticklabels=source_sentence.split(),
                     yticklabels=predicted_sentence, cmap="Greens") # 热力图
    plt.xlabel(" 源序列 ")
    plt.ylabel(" 目标序列 ")
    plt.show() # 显示图片

def test_deq2seq(model, source_sentence):
    encoder_input = np.array([[word2idx_cn[n] for n in source_sentence.split()]])
    decoder_input = np.array([word2idx_en['<sos>']] + [word2idx_en['<eos>']]*(len(encoder_input[0]) - 1))
    encoder_input = torch.LongTensor(encoder_input)
    decoder_input = torch.LongTensor(decoder_input).unsqueeze(0)
    hidden = torch.zeros(1, encoder_input.size(0), n_hidden)
    predict, attn_weights = model(encoder_input, hidden, decoder_input)
    predict = predict.data.max(2, keepdim=True)[1]
    print(source_sentence, '->', [idx2word_en[n.item()] for n in predict.squeeze()])
    attn_weights = attn_weights.squeeze(0).cpu().detach().numpy()
    visualize_attention(source_sentence, [idx2word_en[n.item()] for n in predict.squeeze()], attn_weights)

test_deq2seq(model, '咖哥 喜欢 小冰')
test_deq2seq(model, '自然 语言 处理 很 强大')

