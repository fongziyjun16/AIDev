import numpy as np # 导入 numpy 库
import torch # 导入 torch 库
import torch.nn as nn # 导入 torch.nn 库
d_k = 64 # K(=Q) 维度
d_v = 64 # V 维度
# 定义缩放点积注意力类
class ScaledDotProductAttention(nn.Module):
    def __init__(self):
        super(ScaledDotProductAttention, self).__init__()
    def forward(self, Q, K, V, attn_mask):
        #------------------------- 维度信息 --------------------------------
        # Q K V [batch_size, n_heads, len_q/k/v, dim_q=k/v] (dim_q=dim_k)
        # attn_mask [batch_size, n_heads, len_q, len_k]
        #----------------------------------------------------------------
        # 计算注意力分数（原始权重）[batch_size，n_heads，len_q，len_k]
        scores = torch.matmul(Q, K.transpose(-1, -2)) / np.sqrt(d_k)
        #------------------------- 维度信息 --------------------------------
        # scores [batch_size, n_heads, len_q, len_k]
        #-----------------------------------------------------------------
        # 使用注意力掩码，将 attn_mask 中值为 1 的位置的权重替换为极小值
        #------------------------- 维度信息 --------------------------------
        # attn_mask [batch_size, n_heads, len_q, len_k], 形状和 scores 相同
        #-----------------------------------------------------------------
        scores.masked_fill_(attn_mask, -1e9)
        # 对注意力分数进行 softmax 归一化
        weights = nn.Softmax(dim=-1)(scores)
        #------------------------- 维度信息 --------------------------------
        # weights [batch_size, n_heads, len_q, len_k], 形状和 scores 相同
        #-----------------------------------------------------------------
        # 计算上下文向量（也就是注意力的输出）, 是上下文信息的紧凑表示
        context = torch.matmul(weights, V)
        #------------------------- 维度信息 --------------------------------
        # context [batch_size, n_heads, len_q, dim_v]
        #-----------------------------------------------------------------
        return context, weights # 返回上下文向量和注意力分数

# 定义多头自注意力类
d_embedding = 512  # Embedding 的维度
n_heads = 8  # Multi-Head Attention 中头的个数
batch_size = 3 # 每一批的数据大小
class MultiHeadAttention(nn.Module):
    def __init__(self):
        super(MultiHeadAttention, self).__init__()
        self.W_Q = nn.Linear(d_embedding, d_k * n_heads) # Q的线性变换层
        self.W_K = nn.Linear(d_embedding, d_k * n_heads) # K的线性变换层
        self.W_V = nn.Linear(d_embedding, d_v * n_heads) # V的线性变换层
        self.linear = nn.Linear(n_heads * d_v, d_embedding)
        self.layer_norm = nn.LayerNorm(d_embedding)
    def forward(self, Q, K, V, attn_mask):
        #------------------------- 维度信息 --------------------------------
        # Q K V [batch_size, len_q/k/v, embedding_dim]
        #-----------------------------------------------------------------
        residual, batch_size = Q, Q.size(0) # 保留残差连接
        # 将输入进行线性变换和重塑，以便后续处理
        q_s = self.W_Q(Q).view(batch_size, -1, n_heads, d_k).transpose(1,2)
        k_s = self.W_K(K).view(batch_size, -1, n_heads, d_k).transpose(1,2)
        v_s = self.W_V(V).view(batch_size, -1, n_heads, d_v).transpose(1,2)
        #------------------------- 维度信息 --------------------------------
        # q_s k_s v_s: [batch_size, n_heads, len_q/k/v, d_q=k/v]
        #-----------------------------------------------------------------
        # 将注意力掩码复制到多头 attn_mask: [batch_size, n_heads, len_q, len_k]
        attn_mask = attn_mask.unsqueeze(1).repeat(1, n_heads, 1, 1)
        #------------------------- 维度信息 --------------------------------
        # attn_mask [batch_size, n_heads, len_q, len_k]
        #-----------------------------------------------------------------
        # 使用缩放点积注意力计算上下文和注意力权重
        context, weights = ScaledDotProductAttention()(q_s, k_s, v_s, attn_mask)
        #------------------------- 维度信息 --------------------------------
        # context [batch_size, n_heads, len_q, dim_v]
        # weights [batch_size, n_heads, len_q, len_k]
        #-----------------------------------------------------------------
        # 通过调整维度将多个头的上下文向量连接在一起
        context = context.transpose(1, 2).contiguous().view(batch_size, -1, n_heads * d_v)
        #------------------------- 维度信息 --------------------------------
        # context [batch_size, len_q, n_heads * dim_v]
        #-----------------------------------------------------------------
        # 用一个线性层把连接后的多头自注意力结果转换，原始地嵌入维度
        output = self.linear(context)
        #------------------------- 维度信息 --------------------------------
        # output [batch_size, len_q, embedding_dim]
        #-----------------------------------------------------------------
        # 与输入 (Q) 进行残差链接，并进行层归一化后输出
        output = self.layer_norm(output + residual)
        #------------------------- 维度信息 --------------------------------
        # output [batch_size, len_q, embedding_dim]
        #-----------------------------------------------------------------
        return output, weights # 返回层归一化的输出和注意力权重

# 定义逐位置前馈网络类
class PoswiseFeedForwardNet(nn.Module):
    def __init__(self, d_ff=2048):
        super(PoswiseFeedForwardNet, self).__init__()
        # 定义一维卷积层 1，用于将输入映射到更高维度
        self.conv1 = nn.Conv1d(in_channels=d_embedding, out_channels=d_ff, kernel_size=1)
        # 定义一维卷积层 2，用于将输入映射回原始维度
        self.conv2 = nn.Conv1d(in_channels=d_ff, out_channels=d_embedding, kernel_size=1)
        # 定义层归一化
        self.layer_norm = nn.LayerNorm(d_embedding)
    def forward(self, inputs):
        #------------------------- 维度信息 --------------------------------
        # inputs [batch_size, len_q, embedding_dim]
        #----------------------------------------------------------------
        residual = inputs  # 保留残差连接
        # 在卷积层 1 后使用 ReLU 激活函数
        output = nn.ReLU()(self.conv1(inputs.transpose(1, 2)))
        #------------------------- 维度信息 --------------------------------
        # output [batch_size, d_ff, len_q]
        #----------------------------------------------------------------
        # 使用卷积层 2 进行降维
        output = self.conv2(output).transpose(1, 2)
        #------------------------- 维度信息 --------------------------------
        # output [batch_size, len_q, embedding_dim]
        #----------------------------------------------------------------
        # 与输入进行残差链接，并进行层归一化
        output = self.layer_norm(output + residual)
        #------------------------- 维度信息 --------------------------------
        # output [batch_size, len_q, embedding_dim]
        #----------------------------------------------------------------
        return output # 返回加入残差连接后层归一化的结果

# 生成后续注意力掩码的函数，用于在多头自注意力计算中忽略未来信息
def get_attn_subsequent_mask(seq):
    #------------------------- 维度信息 --------------------------------
    # seq 的维度是 [batch_size, seq_len(Q)=seq_len(K)]
    #-----------------------------------------------------------------
    # 获取输入序列的形状
    attn_shape = [seq.size(0), seq.size(1), seq.size(1)]
    #------------------------- 维度信息 --------------------------------
    # attn_shape 是一个一维张量 [batch_size, seq_len(Q), seq_len(K)]
    #-----------------------------------------------------------------
    # 使用 numpy 创建一个上三角矩阵（triu = triangle upper）
    subsequent_mask = np.triu(np.ones(attn_shape), k=1)
    #------------------------- 维度信息 --------------------------------
    # subsequent_mask 的维度是 [batch_size, seq_len(Q), seq_len(K)]
    #-----------------------------------------------------------------
    # 将 numpy 数组转换为 PyTorch 张量，并将数据类型设置为 byte（布尔值）
    subsequent_mask = torch.from_numpy(subsequent_mask).bool()
    #------------------------- 维度信息 --------------------------------
    # 返回的 subsequent_mask 的维度是 [batch_size, seq_len(Q), seq_len(K)]
    #-----------------------------------------------------------------
    return subsequent_mask # 返回后续位置的注意力掩码

# 定义解码器层类
class DecoderLayer(nn.Module):
    def __init__(self):
        super(DecoderLayer, self).__init__()
        self.self_attn = MultiHeadAttention()  # 多头自注意力层
        self.feed_forward = PoswiseFeedForwardNet()  # 逐位置前馈网络层
        self.norm1 = nn.LayerNorm(d_embedding)  # 第一个层归一化
        self.norm2 = nn.LayerNorm(d_embedding)  # 第二个层归一化
    def forward(self, dec_inputs, attn_mask=None):
        # 使用多头自注意力处理输入
        attn_output, _ = self.self_attn(dec_inputs, dec_inputs, dec_inputs, attn_mask)
        # 将注意力输出与输入相加并进行第一个层归一化
        norm1_outputs = self.norm1(dec_inputs + attn_output)
        # 将归一化后的输出输入到位置前馈神经网络
        ff_outputs = self.feed_forward(norm1_outputs)
        # 将前馈神经网络输出与第一次归一化后的输出相加并进行第二个层归一化
        dec_outputs = self.norm2(norm1_outputs + ff_outputs)
        return dec_outputs # 返回解码器层输出

#  定义解码器类
n_layers = 6  # 设置 Decoder 的层数
class Decoder(nn.Module):
    def __init__(self, vocab_size, max_seq_len):
        super(Decoder, self).__init__()
        # 词嵌入层（参数为词典维度）
        self.src_emb = nn.Embedding(vocab_size, d_embedding)
        # 位置编码层（参数为序列长度）
        self.pos_emb = nn.Embedding(max_seq_len, d_embedding)
        # 初始化 N 个解码器层
        self.layers = nn.ModuleList([DecoderLayer() for _ in range(n_layers)])
    def forward(self, dec_inputs):
        # 创建位置信息
        positions = torch.arange(len(dec_inputs), device=dec_inputs.device).unsqueeze(-1)
        # 将词嵌入与位置编码相加
        inputs_embedding = self.src_emb(dec_inputs) + self.pos_emb(positions)
        # 生成自注意力掩码
        attn_mask = get_attn_subsequent_mask(inputs_embedding).to(device)
        # 初始化解码器输入，这是第一层解码器层的输入
        dec_outputs =  inputs_embedding
        for layer in self.layers:
            # 将输入数据传递给解码器层，并返回解码器层的输出，作为下一层的输入
            dec_outputs = layer(dec_outputs, attn_mask)
        return dec_outputs # 返回解码器输出

# 定义 GPT 模型
class GPT(nn.Module):
    def __init__(self, vocab_size, max_seq_len):
        super(GPT, self).__init__()
        self.decoder = Decoder(vocab_size, max_seq_len) # 解码器，用于学习文本生成能力
        self.projection = nn.Linear(d_embedding, vocab_size)  # 全连接层，输出预测结果
    def forward(self, dec_inputs):
        dec_outputs = self.decoder(dec_inputs) # 将输入数据传递给解码器
        logits = self.projection(dec_outputs) # 传递给全连接层以生成预测
        return logits # 返回预测结果

# 构建语料库
from collections import Counter
class LanguageCorpus:
    def __init__(self, sentences):
        self.sentences = sentences
        # 计算语言的最大句子长度，并加 2 以容纳特殊符号 <sos> 和 <eos>
        self.seq_len = max([len(sentence.split()) for sentence in sentences]) + 2
        self.vocab = self.create_vocabulary() # 创建源语言和目标语言的词汇表
        self.idx2word = {v: k for k, v in self.vocab.items()} # 创建索引到单词的映射
    def create_vocabulary(self):
        vocab = {'<pad>': 0, '<sos>': 1, '<eos>': 2}
        counter = Counter()
        # 统计语料库的单词频率
        for sentence in self.sentences:
            words = sentence.split()
            counter.update(words)
        # 创建词汇表，并为每个单词分配一个唯一的索引
        for word in counter:
            if word not in vocab:
                vocab[word] = len(vocab)
        return vocab
    def make_batch(self, batch_size, test_batch=False):
        input_batch, output_batch = [], [] # 初始化批数据
        sentence_indices = torch.randperm(len(self.sentences))[:batch_size] # 随机选择句子索引
        for index in sentence_indices:
            sentence = self.sentences[index]
            # 将句子转换为索引序列
            seq = [self.vocab['<sos>']] + [self.vocab[word] for word in sentence.split()] + [self.vocab['<eos>']]
            seq += [self.vocab['<pad>']] * (self.seq_len - len(seq)) # 对序列进行填充
            # 将处理好的序列添加到批次中
            input_batch.append(seq[:-1])
            output_batch.append(seq[1:])
        return torch.LongTensor(input_batch), torch.LongTensor(output_batch)

with open("lang.txt", "r") as file: # 从文件中读入语料
    sentences = [line.strip() for line in file.readlines()]
corpus = LanguageCorpus(sentences) # 创建语料库
vocab_size = len(corpus.vocab) # 词汇表大小
max_seq_len = corpus.seq_len # 最大句子长度（用于设置位置编码）
print(f" 语料库词汇表大小 : {vocab_size}") # 打印词汇表大小
print(f" 最长句子长度 : {max_seq_len}") # 打印最大序列长

import torch.optim as optim # 导入优化器
device = "cuda" if torch.cuda.is_available() else "cpu" # 设置设备
model = GPT(vocab_size, max_seq_len).to(device) # 创建 GPT 模型实例
criterion = nn.CrossEntropyLoss() # 损失函数
optimizer = optim.Adam(model.parameters(), lr=0.0001) # 优化器
epochs = 500 # 训练轮次
for epoch in range(epochs):  # 训练 epochs 轮
    optimizer.zero_grad() # 梯度清零
    inputs, targets = corpus.make_batch(batch_size) # 创建训练数据
    inputs, targets = inputs.to(device), targets.to(device)
    outputs = model(inputs) # 获取模型输出
    loss = criterion(outputs.view(-1, vocab_size), targets.view(-1)) # 计算损失
    if (epoch + 1) % 100 == 0: # 打印损失
        print(f"Epoch: {epoch + 1:04d} cost = {loss:.6f}")
    loss.backward() # 反向传播
    optimizer.step() # 更新参数

# 测试文本生成
def generate_text(model, input_str, max_len=50):
    model.eval()  # 将模型设置为评估（测试）模式，关闭 dropout 和 batch normalization 等训练相关的层
    # 将输入字符串中的每个 token 转换为其在词汇表中的索引
    input_tokens = [corpus.vocab[token] for token in input_str]
    # 创建一个新列表，将输入的 tokens 复制到输出 tokens 中 , 目前只有输入的词
    output_tokens = input_tokens.copy()
    with torch.no_grad():  # 禁用梯度计算，以节省内存并加速测试过程
        for _ in range(max_len):  # 生成最多 max_len 个 tokens
            # 将输出的 token 转换为 PyTorch 张量，并增加一个代表批次的维度 [1, len(output_tokens)]
            inputs = torch.LongTensor(output_tokens).unsqueeze(0).to(device)
            outputs = model(inputs) # 输出 logits 形状为 [1, len(output_tokens), vocab_size]
            # 在最后一个维度上获取 logits 中的最大值，并返回其索引（即下一个 token）
            _, next_token = torch.max(outputs[:, -1, :], dim=-1)
            next_token = next_token.item() # 将张量转换为 Python 整数
            if next_token == corpus.vocab["<eos>"]:
                break # 如果生成的 token 是 EOS（结束符），则停止生成过程
            output_tokens.append(next_token) # 将生成的 tokens 添加到 output_tokens 列表
    # 将输出 tokens 转换回文本字符串
    output_str = " ".join([corpus.idx2word[token] for token in output_tokens])
    return output_str
input_str = ["Python"] # 输入一个词：Python
generated_text = generate_text(model, input_str) # 模型跟着这个词生成后续文本
print(" 生成的文本 :", generated_text) # 打印预测文本
