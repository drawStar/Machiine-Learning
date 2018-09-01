#coding=utf-8
import os
import torch
import torchvision
import torch.utils.data as Data
import matplotlib.pyplot as plt

LR = 0.01
BATCH_SIZE = 64
TIME_STEP=28  #一张图片高度为28，共需要28次
INPUT_SIZE=28 #一张兔皮哦按宽度为28，一次输入为一整行的像素信息
EPOCH = 1
DOWNLOAD_MNIST = False

train_data = torchvision.datasets.MNIST(
    root='./mnist/',
    train=True,                                     # this is training data
    transform=torchvision.transforms.ToTensor(),    # Converts a PIL.Image or numpy.ndarray to
                                                    # torch.FloatTensor of shape (C x H x W) and normalize in the range [0.0, 1.0]
    download=DOWNLOAD_MNIST,
)
train_loader=torch.utils.data.DataLoader(
    dataset=train_data,
    batch_size=BATCH_SIZE,
    shuffle=True,
)
test_data = torchvision.datasets.MNIST(root='./mnist/', train=False)
# test_x=torch.unsqueeze(test_data.test_data,dim=1).type(torch.FloatTensor)[:2000]/255
# # shape from (2000, 28, 28) to (2000, 1, 28, 28), value in range(0,1)
# #/255是为了保证在0~1之间，与train——data相对应
# test_y=test_data.test_labels[:2000]
#????
test_x = test_data.test_data.type(torch.FloatTensor)[:2000]/255.   # shape (2000, 28, 28) value in range(0,1)
test_y = test_data.test_labels.numpy().squeeze()[:2000]    # covert to numpy array

class RNN(torch.nn.Module):
    def __init__(self):
        super(RNN,self).__init__()
        self.rnn=torch.nn.LSTM(
            input_size=INPUT_SIZE,
            hidden_size=64,         # rnn hidden unit
            num_layers=1,           # number of rnn layer
            batch_first=True,       # input & output will has batch size as 1s dimension. e.g. (batch, time_step, input_size)
            #True表示讲batch放在前面
        )
        self.out=torch.nn.Linear(64,10)
    def forward(self, x):
        # x shape (batch, time_step, input_size)
        # rnn_out shape (batch, time_step, output_size)
        # h_n shape (n_layers, batch, hidden_size),“分线程”的hidden state
        # h_c shape (n_layers, batch, hidden_size)，“总线程”的hidden state
        #None表示初始的hidden state，没有为None
        rnn_out, (h_n, h_c) = self.rnn(x, None)  # None represents zero initial hidden state
        #因为rnn_out的shape为 (batch, time_step, output_size)
        # 我们需要选取最后一个时间点的output，-1表示选timestep最后一个时间点
        output=self.out(rnn_out[:,-1,:])
        return output

rnn=RNN()

optimizer=torch.optim.Adam(rnn.parameters(),lr=LR)
loss_func=torch.nn.CrossEntropyLoss()

for epoch in range(EPOCH):
    for step,(b_x,b_y) in enumerate(train_loader):
        b_x=b_x.view(-1,28,28) #   # reshape x to (batch, time_step, input_size)
        output=rnn(b_x)
        loss=loss_func(output,b_y)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if step %50==0:
            test_output= rnn(test_x)
            pred_y=torch.max(test_output,1)[1].data.squeeze().numpy()
            accuracy = float((pred_y == test_y).astype(int).sum()) / float(test_y.size)
            print('Epoch: ', epoch, '| train loss: %.4f' % loss.data.numpy(), '| test accuracy: %.2f' % accuracy)

# print 10 predictions from test data
test_output = rnn(test_x[:10].view(-1, 28, 28))
pred_y = torch.max(test_output, 1)[1].data.numpy().squeeze()
print(pred_y, 'prediction number')
print(test_y[:10], 'real number')
