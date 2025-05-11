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
device = "cuda" if torch.cuda.is_available() else "cpu"  # 设置设备
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


from torchtext.datasets import WikiText2 # 导入WikiText2
from torchtext.data.utils import get_tokenizer # 导入Tokenizer分词工具
from torchtext.vocab import build_vocab_from_iterator # 导入Vocabulary工具
from torch.utils.data import DataLoader, Dataset # 导入Pytorch的DataLoader和Dataset

tokenizer = get_tokenizer("basic_english") # 定义数据预处理所需的tokenizer

# train_iter = WikiText2(split="train") # 加载WikiText2数据集的训练部分
train_tokens_filepath = "./datasets/WikiText2/wikitext-2/wiki.train.tokens" # 加载WikiText2数据集的训练部分
# valid_iter = WikiText2(split="valid") # 加载WikiText2数据集的验证部分
valid_tokens_filepath = "./datasets/WikiText2/wikitext-2/wiki.valid.tokens" # 加载WikiText2数据集的训练部分

# 定义一个生成器函数，用于将数据集中的文本转换为tokens
def yield_tokens(data_iter):
    for item in data_iter:
        yield tokenizer(item)

def yield_tokens_from_file(filepath):
    with open(filepath, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                yield tokenizer(line)

# 创建词汇表，包括特殊tokens："<pad>", "<sos>", "<eos>"
vocab = build_vocab_from_iterator(yield_tokens_from_file(train_tokens_filepath), specials=["<pad>", "<sos>", "<eos>"])
vocab.set_default_index(vocab["<pad>"])

# 打印词汇表信息
print("词汇表大小:", len(vocab))
print("词汇示例(word to index):",
      {word: vocab[word] for word in ["<pad>", "<sos>", "<eos>", "the", "apple"]})

max_seq_len = 256  # 设置序列的最大长度

# 定义一个处理WikiText2数据集的自定义数据集类
class WikiDataset(Dataset):
    def __init__(self, data_iter, vocab, max_len=max_seq_len):
        self.data = []
        for sentence in data_iter:  # 遍历数据集，将文本转换为tokens
            # 对每个句子进行tokenization，并截取长度为max_len-2，为<sos>和<eos>留出空间
            tokens = tokenizer(sentence)[:max_len - 2]
            tokens = [vocab["<sos>"]] + vocab(tokens) + [vocab["<eos>"]]  # 添加<sos>和<eos>
            self.data.append(tokens)  # 将处理好的tokens添加到数据集中

    def __len__(self):  # 定义数据集的长度
        return len(self.data)

    def __getitem__(self, idx):  # 定义数据集的索引方法 (即抽取数据条目)
        source = self.data[idx][:-1]  # 获取当前数据，并将<eos>移除，作为source
        target = self.data[idx][1:]  # 获取当前数据，并将<sos>移除，作为target（右移1位）
        return torch.tensor(source), torch.tensor(target)  # 转换为tensor并返回

def read_lines(filepath):
    with open(filepath, encoding='utf-8') as f:
        for line in f:
            yield line

# train_dataset = WikiDataset(train_iter, vocab)  # 创建训练数据集
train_dataset = WikiDataset(read_lines(train_tokens_filepath), vocab)  # 创建训练数据集
# valid_dataset = WikiDataset(valid_iter, vocab)  # 创建验证数据集
valid_dataset = WikiDataset(read_lines(valid_tokens_filepath), vocab) # 创建验证数据集
print(f"Dataset数据条目: {len(train_dataset)}")
sample_source, sample_target = train_dataset[100]
print(f"输入序列张量样例: {sample_source}")
print(f"目标序列张量样例: {sample_target}")
decoded_source = ' '.join(vocab.lookup_tokens(sample_source.tolist()))
decoded_target = ' '.join(vocab.lookup_tokens(sample_target.tolist()))
print(f"输入序列样例文本: {decoded_source}")
print(f"目标序列样例文本: {decoded_target}")

# 定义pad_sequence函数，用于将一批序列补齐到相同长度
def pad_sequence(sequences, padding_value=0, length=None):
    # 计算最大序列长度，如果length参数未提供，则使用输入序列中的最大长度
    max_length = max(len(seq) for seq in sequences) if length is None else length
    # 创建一个具有适当形状的全零张量，用于存储补齐后的序列
    result = torch.full((len(sequences), max_length), padding_value, dtype=torch.long)
    # 遍历序列，将每个序列的内容复制到结果张量中
    for i, seq in enumerate(sequences):
        end = len(seq)
        result[i, :end] = seq[:end]
    return result

# 定义collate_fn函数，用于将一个批次的数据整理成适当的形状
def collate_fn(batch):
    # 从批次中分离源序列和目标序列
    sources, targets = zip(*batch)
    # 计算批次中的最大序列长度
    max_length = max(max(len(s) for s in sources), max(len(t) for t in targets))
    # 使用pad_sequence函数补齐源序列和目标序列
    sources = pad_sequence(sources, padding_value=vocab["<pad>"], length=max_length)
    targets = pad_sequence(targets, padding_value=vocab["<pad>"], length=max_length)
    # 返回补齐后的源序列和目标序列
    return sources, targets

# 创建一个训练数据加载器，使用自定义的collate_fn函数
train_dataloader = DataLoader(train_dataset, batch_size=batch_size,
                              shuffle=True, collate_fn=collate_fn)
# 创建一个验证数据加载器，使用自定义的collate_fn函数
valid_dataloader = DataLoader(valid_dataset, batch_size=batch_size,
                              shuffle=False, collate_fn=collate_fn)

import torch.optim as optim  # 导入优化器
device = "cuda" if torch.cuda.is_available() else "cpu"  # 设置设备
model = GPT(len(vocab), max_seq_len).to(device)  # 创建GPT模型实例
criterion = nn.CrossEntropyLoss(ignore_index=vocab["<pad>"])
optimizer = optim.Adam(model.parameters(), lr=0.0001)  # 优化器
epochs = 2  # 训练轮次

import time

# total_time = 0
# count = 0
# for epoch in range(epochs):
#     epoch_loss = 0
#     for batch_idx, (source, target) in enumerate(train_dataloader): # 用Dataloader加载数据
#         st = time.time()
#         inputs, targets = source.to(device), target.to(device)
#         optimizer.zero_grad()  # 梯度清零
#         outputs = model(inputs)  # 获取模型输出
#         loss = criterion(outputs.view(-1, len(vocab)), targets.view(-1))  # 计算损失
#         loss.backward()  # 反向传播
#         optimizer.step()  # 更新参数
#         epoch_loss += loss.item()
#         if (batch_idx + 1) % 500 == 0: # 每500个批次打印一次损失
#             print(f"Batch {batch_idx + 1}/{len(train_dataloader)}, Loss: {loss.item()}")
#         et = time.time()
#         total_time = total_time + (et -st)
#         count = count + 1
#         if (batch_idx + 1) % 100 == 0:  # 每100个批次打印一次平均batch时间
#             print(f"one batch uses avg {total_time / count} s")
#     epoch_loss /= len(train_dataloader) # 每轮打印一次损失
#     print(f"Epoch {epoch + 1}/{epochs}, Average Loss: {epoch_loss}")

from datetime import datetime

# Save the trained model
# timestamp = datetime.fromtimestamp(time.time()).strftime('%Y-%m-%d_%H-%M-%S')
# model_file_name = f"trained_model_{timestamp}.pt"
# torch.save(model.state_dict(), model_file_name)
# print(f"Model saved as {model_file_name}")

# Replace 'model_timestamp.pt' with your saved model's filename
model.load_state_dict(torch.load('trained_model_2025-05-12_00-42-34.pt'))
# 测试文本生成
def generate_text_greedy_search(model, input_str, max_len=50):
    model.eval()  # 将模型设置为评估（测试）模式，关闭dropout和batch normalization等训练相关的层
    # 将输入字符串中的每个Token 转换为其在词汇表中的索引
    input_tokens = [vocab[token] for token in input_str.split()]
    # 创建一个新列表，将输入的Token复制到输出Token中,目前只有输入的词
    output_tokens = input_tokens.copy()
    with torch.no_grad():  # 禁用梯度计算，以节省内存并加速测试过程
        for _ in range(max_len):  # 生成最多max_len个Token
            # 将输出token转换为 PyTorch张量，并增加一个代表批次的维度[1, len(output_tokens)]
            inputs = torch.LongTensor(output_tokens).unsqueeze(0).to(device)
            outputs = model(inputs) # 输出 logits形状为[1, len(output_tokens), vocab_size]
            logits = outputs[:, -1, :] # 只关心最后一个时间步（即最新生成的token）的logits
            # 在最后一个维度上获取logits中的最大值，并返回其索引（即下一个Token）
            _, next_token = torch.max(logits, dim=-1)
            next_token = next_token.item() # 将张量转换为Python整数
            if next_token == vocab["<eos>"]:
                break # 如果生成的Token是 EOS（结束符），则停止生成过程
            output_tokens.append(next_token) # 将生成的Token添加到output_tokens列表
    # 将输出Token转换回文本字符串
    output_str = " ".join([vocab.get_itos()[token] for token in output_tokens
                           if vocab.get_itos()[token] != "<pad>" and vocab.get_itos()[token] != "<unk>" ])
    return output_str

input_str = "how are you" # 输入一个词：Python
generated_text = generate_text_greedy_search(model, input_str) # 模型跟着这个字生成后续文本
print("生成的文本:", generated_text) # 打印预测文本

# 定义集束搜索的函数
def generate_text_beam_search(model, input_str, max_len=50, beam_width=5):
    model.eval()  # 将模型设置为评估（测试）模式，关闭dropout和batch normalization等训练相关的层
    # 将输入字符串中的每个token 转换为其在词汇表中的索引
    input_tokens = [vocab[token] for token in input_str.split()]
    # 创建一个列表，用于存储候选序列
    candidates = [(input_tokens, 0.0)]
    with torch.no_grad():  # 禁用梯度计算，以节省内存并加速测试过程
        for _ in range(max_len):  # 生成最多max_len个tokens
            new_candidates = []
            for candidate, candidate_score in candidates:
                inputs = torch.LongTensor(candidate).unsqueeze(0).to(device)
                outputs = model(inputs) # 输出 logits形状为[1, len(output_tokens), vocab_size]
                logits = outputs[:, -1, :] # 只关心最后一个时间步（即最新生成的token）的logits
                # 找到具有最高分数的前beam_width个tokens
                scores, next_tokens = torch.topk(logits, beam_width, dim=-1)
                final_results = [] # 初始化输出序列
                for score, next_token in zip(scores.squeeze(), next_tokens.squeeze()):
                    new_candidate = candidate + [next_token.item()]
                    new_score = candidate_score - score.item()  # 使用负数，因为我们需要降序排列
                    if next_token.item() == vocab["<eos>"]:
                        # 如果生成的token是EOS（结束符），将其添加到最终结果中
                        final_results.append((new_candidate, new_score))
                    else:
                        # 将新生成的候选序列添加到新候选列表中
                        new_candidates.append((new_candidate, new_score))
            # 从新候选列表中选择得分最高的beam_width个序列
            candidates = sorted(new_candidates, key=lambda x: x[1])[:beam_width]
    # 选择得分最高的候选序列
    best_candidate, _ = sorted(candidates, key=lambda x: x[1])[0]
    # 将输出 token 转换回文本字符串
    output_str = " ".join([vocab.get_itos()[token] for token in best_candidate if vocab.get_itos()[token] != "<pad>"])
    return output_str

input_str = "my name"  # 输入几个词
generated_text = generate_text_beam_search(model, input_str)  # 模型跟着这些词生成后续文本
print("生成的文本:", generated_text)  # 打印生成的文本
