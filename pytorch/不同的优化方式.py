#coding=utf-8
import torch
import torch.utils.data as Data
import torch.nn.functional as F
import matplotlib.pyplot as plt

LR = 0.01
BATCH_SIZE = 32
EPOCH = 12

x=torch.unsqueeze(torch.linspace(-1,1,1000),dim=1)
y=x.pow(2)+0.1*torch.normal(torch.zeros(*x.size()))

# plt.scatter(x.data.numpy(),y.data.numpy())
# plt.show()

torch_dataset=Data.TensorDataset(x,y)
loader=Data.DataLoader(
    dataset=torch_dataset,
    batch_size=BATCH_SIZE,
    shuffle=True,num_workers=2
)

class Net(torch.nn.Module):
    def __init__(self):
        super(Net,self).__init__()
        self.hidden=torch.nn.Linear(1,20)
        self.out=torch.nn.Linear(20,1)

    def forward(self,x):
        x=self.hidden(x)
        x=F.relu(x)
        x=self.out(x)
        return x

if __name__ == '__main__':
    netSGD=Net()
    netMomentum=Net()
    netRMSprop = Net()
    netAdam = Net()
    nets=[netSGD,netMomentum,netRMSprop,netAdam]#组成一个list

    opt_SGD=torch.optim.SGD(netSGD.parameters(),lr=LR)
    opt_Momentum = torch.optim.SGD(netMomentum.parameters(), lr=LR,momentum=0.8)
    opt_RMSprop = torch.optim.RMSprop(netRMSprop.parameters(), lr=LR,alpha=0.9)
    opt_Adam = torch.optim.Adam(netAdam.parameters(), lr=LR,betas=(0.9,0.99))
    optimizers = [opt_SGD, opt_Momentum, opt_RMSprop, opt_Adam]

    loss_func=torch.nn.MSELoss()
    losses=[[],[],[],[]]

    for epoch in range(EPOCH):
        for step,(b_x,b_y) in enumerate(loader):
            for net,opt,l in zip(nets,optimizers,losses):
                output=net(b_x)
                loss=loss_func(output,b_y)
                opt.zero_grad()
                loss.backward()
                opt.step()
                l.append(loss.data.numpy())

    labels = ['SGD', 'Momentum', 'RMSprop', 'Adam']
    for i, l_his in enumerate(losses):
        plt.plot(l_his, label=labels[i])
    plt.legend(loc='best')
    plt.xlabel('Steps')
    plt.ylabel('Loss')
    plt.ylim((0, 0.2))
    plt.show()
