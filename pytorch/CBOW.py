import torch
import torch.autograd as autograd
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

torch.manual_seed(1)

"""
CBOW模型，输入是context的index，获得对应那一行的随机初始化的embedding,再将context embedding做个sum
最后一层输出的尺寸为vocab_size(即要学习每个word的embedding),label为targetword
NLLLoss（logprob,target）为损失函数
"""


raw_text = """We are about to study the idea of a computational process.
Computational processes are abstract beings that inhabit computers.
As they evolve, processes manipulate other abstract things called data.
The evolution of a process is directed by a pattern of rules
called a program. People create programs to direct processes. In effect,
we conjure the spirits of the computer with our spells.""".split()


data = []
for i in range(2, len(raw_text)-2):
    context = [raw_text[i -2], raw_text[i -1], raw_text[i + 1], raw_text[i + 2]]
    target = raw_text[i]
    data.append((context, target))

word_to_id ={}
for word in raw_text:
    if word not in word_to_id:
        word_to_id[word] = len(word_to_id)

id_to_word = {v:k for k, v in word_to_id.items()}

VOCAB_SIZE = len(word_to_id)
CONTEXT_SIZE = 2  # 2 words to the left, 2 to the right
EMBEDDING_DIM = 10
EPOCHS = 150

class CBOW(nn.Module):
    def __init__(self, vocab_size, embedding_dim, context_size):
        super(CBOW, self).__init__()
        self.embeddings = nn.Embedding(vocab_size, embedding_dim)
        self.linear1 = nn.Linear(embedding_dim, 128)
        self.linear2 = nn.Linear(128, vocab_size)
    def forward(self, x):
        # TODO:为什么view((1, -1))
        embeds = sum(self.embeddings(x)).view((1, -1))
        t1 = F.relu(self.linear1(embeds))
        t2 = self.linear2(t1)
        out = F.log_softmax(t2, dim =1)
        return out

def make_context_tensor(context, word_to_id):
    context_idx = torch.LongTensor([word_to_id[word] for word in context])
    return context_idx
losses = []# 记录训练的loss
loss_function = nn.NLLLoss()
model = CBOW(VOCAB_SIZE, EMBEDDING_DIM, CONTEXT_SIZE)
optimizer = optim.SGD(model.parameters(), lr=0.001)

def train():
    model.train()
    for epoch in range(EPOCHS):
        total_loss = 0
        # TODO: 输入是什么?
        for context, targetword in data:
            #获得contextindex行的embedding向量，别忘了转为tensor类型
            context_index = make_context_tensor(context,word_to_id)
            target = torch.LongTensor([word_to_id[targetword]])
            model.zero_grad()
            log_probs = model(context_index)
            loss = loss_function(log_probs, target)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
        # losses.append(total_loss)
        print(total_loss)
    print("train done")

def test(testcontext):
    model.eval()
    with torch.no_grad():
        context_tensor = make_context_tensor(testcontext, word_to_id)
        log_probs = model(context_tensor)

        print(log_probs, end='==')  # print 不换行
        # TODO: 选择log_prob 大的为label
        print(id_to_word[log_probs.argmax().item()])

train()
testcontext = ['People','create','to', 'direct']
test(testcontext)
#TODO:获取单词的embedding


