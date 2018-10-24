import torch
import torch.autograd as autograd
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

torch.manual_seed(1)
"""
词性标注， 学习使用LSTM.

"""



training_data = [
    ("The dog ate the apple".split(), ["DET", "NN", "V", "DET", "NN"]),
    ("Everybody read that book".split(), ["NN", "V", "DET", "NN"])
]

# print(training_data)

word_to_id = {}
for sentence, tags in training_data:
    for word in sentence:
        if word not in word_to_id:
            word_to_id[word] = len(word_to_id)
tag_to_id ={"DET": 0, "NN": 1, "V": 2}
id_to_tag = {v:k for k, v in tag_to_id.items()}


def make_tensor(sequence, word_to_id):
    idx = torch.LongTensor([word_to_id[w] for w in sequence])
    return idx

EMBEDDING_DIM = 6
HIDDEN_DIM = 6
EPOCHS = 300


class LSTMTagger(nn.Module):

    def __init__(self, embedding_dim, hidden_dim, vocab_size, target_size):
        super(LSTMTagger, self).__init__()
        self.embeddings = nn.Embedding(vocab_size, embedding_dim)
        self.initial_hidden = self.init_hidden(hidden_dim)
        self.lstm = nn.LSTM(input_size=embedding_dim, hidden_size=hidden_dim)
        self.linear1 = nn.Linear(hidden_dim, target_size)

    def init_hidden(self, hidden_dim):
        h0 = torch.zeros(1, 1, hidden_dim)
        c0 = torch.zeros(1, 1, hidden_dim)
        return h0, c0

    def forward(self, x):
        # x is a sentence
        embeds = self.embeddings(x)
        lstm_out, hidden = self.lstm(embeds.view(len(x), 1, -1), self.initial_hidden)
        # lstm_out shape: [seq_len, batch_size, hidden_size]
        linear1 =self.linear1(lstm_out.view(len(x), -1))
        out = F.log_softmax(linear1, dim=1)
        return out


model = LSTMTagger(EMBEDDING_DIM, HIDDEN_DIM, len(word_to_id), len(tag_to_id))
loss_func = nn.NLLLoss()
optimizer = optim.SGD(model.parameters(), lr=0.1)


def train():
    model.train()
    for epoch in range(EPOCHS):
        total_loss = 0
        for sentence, tags in training_data:
            sentence_idx = make_tensor(sentence, word_to_id)
            target = make_tensor(tags, tag_to_id)

            model.zero_grad()
            # model.init_hidden(HIDDEN_DIM)#TODO：不明白这步的作用，
            logprobs = model(sentence_idx)
            loss = loss_func(logprobs, target)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        print(total_loss)

def test(testsentence):
    model.eval()
    with torch.no_grad():
        testsentence_idx = make_tensor(testsentence, word_to_id)
        log_probs = model(testsentence_idx)
        print(log_probs)
        # TODO: 选择log_prob 大的label
        for i in range(len(testsentence)):
            print(id_to_tag[log_probs[i].argmax().item()])

train()
testsentence="the dog ate the apple".split()
test(testsentence)