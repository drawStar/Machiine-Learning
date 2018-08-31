#coding=utf-8
import torch
import torch.utils.data as Data

BATCH_SIZE=5#不能被整除的，剩下几个训练几个，比如batchsize=8，最后一次训练2个数据

x=torch.linspace(1,10,10)
y=torch.linspace(10,1,10)

torch_dataset=Data.TensorDataset(x,y)
loader=Data.DataLoader(
    dataset=torch_dataset,
    batch_size=BATCH_SIZE,
    shuffle=True,#如果不打乱，每个epoch训练的数据是一样的
    num_workers=2#用于数据加载的子进程数
)

def show_batch():
    for epoch in range(3):
        for step,(batch_x,batch_y) in enumerate(loader):
            print('Epoch: ', epoch, '| Step: ', step, '| batch x: ',
                  batch_x.numpy(), '| batch y: ', batch_y.numpy())



if __name__ == '__main__':
    show_batch()

