import torch
import torch.autograd as autograd
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim


"""
BOW词袋模型
the BOW vector is [Count(word1),Count(word2),Count(word3)...]
the output is logSoftmax(Ax+b)
首先建立词典，为给语料库中的每个word分配一个ID；
统计每句话的词频，建立BoW vector，计算logprob=logSoftmax(Ax+b)；
转换target的int类型，计算损失函数NLLLoss，公式官方文档有）
反向传播。。。
"""
data = [("me gusta comer en la cafeteria".split(), "SPANISH"),
        ("Give it to me".split(), "ENGLISH"),
        ("No creo que sea una buena idea".split(), "SPANISH"),
        ("No it is not a good idea to get lost at sea".split(), "ENGLISH")]

test_data = [("Yo creo que si".split(), "SPANISH"),
             ("it is lost on me".split(), "ENGLISH")]
# print(data + test_data)

# 建立字典，给语料库中的每个word分配一个ID，如{'me': 0, 'gusta': 1, 'comer': 2, ...}
word_to_id = {}
label_to_id = {"SPANISH": 0, "ENGLISH": 1}
id_to_label = {v:k for k, v in label_to_id.items() }#反转字典，在test时输出label
for sentence, label in data + test_data:
    for word in sentence:
        if word not in word_to_id:
            word_to_id[word] = len(word_to_id)
# print(word_to_id)

VOCAB_SIZE = len(word_to_id)
NUM_LABELS = 2
EPOCHS = 50

class BowClassifier(nn.Module):
    def __init__(self, NUM_LABELS, VOCAB_SIZE):
        super(BowClassifier, self).__init__()
        # Ax+b通过nn.Liner()实现
        self.linear = nn.Linear(VOCAB_SIZE, NUM_LABELS)
        # TODO: input为什么是VOCAB_SIZE
    def forward(self, x):
        t1=self.linear(x)
        # t1 size [1,2]
        #TODO:dim?
        out = F.log_softmax(t1, dim=1)
        return out

def makeBowVector(sentence, word_to_id):
    # 统计每句话中的词频
    vec = torch.zeros(VOCAB_SIZE)
    for word in sentence:
        vec[word_to_id[word]] += 1
    return vec.view(1, -1)

def make_target(label, label_to_id):
    # 注意需要将target转变为LongTensor，因为input为floattensor，使用intTensor会报错.
    # To create a tensor of integer types, try torch.LongTensor().
    return torch.LongTensor([label_to_id[label]]) #注意有个[]


model = BowClassifier(NUM_LABELS, VOCAB_SIZE)

# NLLLossd（input,target）input为（N，C）C为类别总数，具体看官方文档
loss_function = nn.NLLLoss()
optimizer = optim.SGD(model.parameters(), lr=0.1)

#train
def train():
    model.train()
    for epoch in range(EPOCHS):
        for sentence, label in data:
            model.zero_grad()
            bow_vec = makeBowVector(sentence, word_to_id)
            target = make_target(label, label_to_id)
            log_probs = model(bow_vec)
            loss = loss_function(log_probs, target)
            loss.backward()
            optimizer.step()
    print("train done")

def test():
    model.eval()
    with torch.no_grad():
        for sentence, label in test_data:
            bow_vec = makeBowVector(sentence, word_to_id)
            log_probs = model(bow_vec)
            print(log_probs, end='==')# print 不换行
            # TODO: 选择log_prob 大的为label
            print(id_to_label[log_probs.argmax().item()])



train()
test()
