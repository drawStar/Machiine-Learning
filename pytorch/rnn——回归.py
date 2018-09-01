#如何用hidden_state作为输入
#用sin去预测cos

#coding=utf-8
import os
import torch
import numpy as np
import matplotlib.pyplot as plt

LR = 0.02
TIME_STEP=10  #一张图片高度为28，共需要28次
INPUT_SIZE=1 #一张兔皮哦按宽度为28，一次输入为一整行的像素信息


# steps=np.linspace(0,np.pi*2,100,dtype=np.float32)
# x_np=np.sin(steps)
# y_np=np.cos(steps)
# plt.plot(steps,y_np,'r-',label='target(cos)')
# plt.plot(steps,x_np,'b-',label='input(cos)')
# plt.legend(loc='best')
# plt.show()
class RNN(torch.nn.Module):
    def __init__(self):
        super(RNN, self).__init__()

        self.rnn = torch.nn.RNN(
            input_size=INPUT_SIZE,
            hidden_size=32,     # rnn hidden unit
            num_layers=1,       # number of rnn layer
            batch_first=True,   # input & output will has batch size as 1s dimension. e.g. (batch, time_step, input_size)
        )
        self.out = torch.nn.Linear(32, 1)
    def forward(self, x, h_state):
        # x shape (batch, time_step, input_size)
        # h_state (n_layers, batch, hidden_size)
        # rnn_out shape (batch, time_step, output_size)

        rnn_output,h_state = self.rnn(x,h_state)
        # rnn_output = rnn_output.view(-1, 32)
        # outs = self.out(rnn_output)
        # return outs, h_state
        outs=[]        #记录每一timestep的output
        for time_step in range(rnn_output.size(1)):
            outs.append(self.out(rnn_output[:,time_step,:]))
        # #因为outs是list 的形式，需要将每一个time——step的out组合在一起
        return torch.stack(outs,dim=1),h_state
rnn=RNN()
optimizer=torch.optim.Adam(rnn.parameters(),lr=LR)
loss_func=torch.nn.MSELoss()
h_state=None#初始的h——state为None

plt.figure(1, figsize=(12, 5))
plt.ion()           # continuously plot

for step in range(100):
    start, end = step * np.pi, (step+1)*np.pi   # time range
    # use sin predicts cos
    steps = np.linspace(start, end, TIME_STEP, dtype=np.float32)
    x_np = np.sin(steps)    # float32 for converting torch FloatTensor
    y_np = np.cos(steps)
#将numpy.ndarray 转换为Tensor,np.newaxis为增加一个维度为1的维度
    x = torch.from_numpy(x_np[np.newaxis, :, np.newaxis])    # shape (batch, time_step, input_size)
    y = torch.from_numpy(y_np[np.newaxis, :, np.newaxis])

    prediction, h_state = rnn(x, h_state)   # rnn output
    # !! next step is important !!
    h_state = h_state.data        # repack the hidden state, break the connection from last iteration

    loss = loss_func(prediction, y)         # calculate loss
    optimizer.zero_grad()                   # clear gradients for this training step
    loss.backward()                         # backpropagation, compute gradients
    optimizer.step()                        # apply gradients


# plotting
    plt.plot(steps, y_np.flatten(), 'r-')
    plt.plot(steps, prediction.data.numpy().flatten(), 'b-')
    plt.draw(); plt.pause(0.05)

plt.ioff()
plt.show()
