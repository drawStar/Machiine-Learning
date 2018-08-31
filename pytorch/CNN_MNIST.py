#coding=utf-8
import os
import torch
import torchvision
import torch.utils.data as Data
import matplotlib.pyplot as plt

LR = 0.01
BATCH_SIZE = 50
EPOCH = 1
DOWNLOAD_MNIST = False

if not(os.path.exists('./mnist/')) or not os.listdir('./mnist/'):
    # not mnist dir or mnist is empyt dir
    DOWNLOAD_MNIST = True

train_data = torchvision.datasets.MNIST(
    root='./mnist/',
    train=True,                                     # this is training data
    transform=torchvision.transforms.ToTensor(),    # Converts a PIL.Image or numpy.ndarray to
                                                    # torch.FloatTensor of shape (C x H x W) and normalize in the range [0.0, 1.0]
    download=DOWNLOAD_MNIST,
)

# Data Loader for easy mini-batch return in training, the image batch shape will be (50, 1, 28, 28)
train_loader=Data.DataLoader(
    dataset=train_data,
    batch_size=BATCH_SIZE,
    shuffle=True,
)
test_data = torchvision.datasets.MNIST(root='./mnist/', train=False)
test_x=torch.unsqueeze(test_data.test_data,dim=1).type(torch.FloatTensor)[:2000]/255
# shape from (2000, 28, 28) to (2000, 1, 28, 28), value in range(0,1)
#/255是为了保证在0~1之间，与train——data相对应
test_y=test_data.test_labels[:2000]

class CNNNet(torch.nn.Module):
    def __init__(self):
        super(CNNNet,self).__init__()
        self.conv1=torch.nn.Sequential( # input shape (1, 28, 28)
            torch.nn.Conv2d(
                in_channels=1,  # input height
                out_channels=16,  # n_filters
                kernel_size=5,  # filter size
                stride=1,  # filter movement/step
                padding=2,
                # if want same width and length of this image after con2d, padding=(kernel_size-1)/2 if stride=1
            ), # input shape (16, 28, 28)
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(kernel_size=2)
        )
        self.conv2=torch.nn.Sequential( # input shape (16, 14, 14)
            torch.nn.Conv2d(
                in_channels=16,
                out_channels=32,
                kernel_size=5,
                stride=1,
                padding=2
            ),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(kernel_size=2)
        ) # input shape (32, 7, 7)
        self.out=torch.nn.Linear(32*7*7,10)#全连接

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)# shape (batch,32,7,7)
        x = x.view(x.size(0),-1)#reshape 为(batch,32*7*7)以便后续全连接层计算
        output=self.out(x)
        return output,x ###?# return x for visualization

cnn=CNNNet()
optimizer=torch.optim.Adam(cnn.parameters(),lr=LR)
loss_func=torch.nn.CrossEntropyLoss()

for epoch in range(EPOCH):
    for step, (b_x, b_y) in enumerate(train_loader):   # gives batch data, normalize x when iterate train_loader
        output = cnn(b_x)[0]               # [0]???
        loss = loss_func(output, b_y)   # cross entropy loss
        optimizer.zero_grad()           # clear gradients for this training step
        loss.backward()                 # backpropagation, compute gradients
        optimizer.step()                # apply gradients

        if step %50==0:
            test_output, last_layer = cnn(test_x)
            pred_y=torch.max(test_output,1)[1].data.squeeze().numpy()
            accuracy = float((pred_y == test_y.data.numpy()).astype(int).sum()) / float(test_y.size(0))
            print('Epoch: ', epoch, '| train loss: %.4f' % loss.data.numpy(), '| test accuracy: %.2f' % accuracy)

# print 10 predictions from test data
test_output, _ = cnn(test_x[:10])
pred_y = torch.max(test_output, 1)[1].data.numpy().squeeze()
print(pred_y, 'prediction number')
print(test_y[:10].numpy(), 'real number')
