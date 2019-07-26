import numpy as np
import matplotlib.pyplot as plt

"""
数据样本格式为[x1 x2 label]
sigmoid(W^T * X)
要训练的是W
对于线性边界： w0x0（x0为1）+w1x1+w2x2+...wnxn

https://blog.csdn.net/achuo/article/details/51160101#commentBox
"""


def loadDataSet(fileName):
    dataList = []
    labelList = []
    fr = open(fileName)
    for line in fr.readlines():
        lineArr = line.strip().split()

        # 为了方便计算，我们将 X0 的值设为 1.0 ，也就是在每一行的开头添加一个 1.0 作为 X0
        dataList.append([1.0, float(lineArr[0]), float(lineArr[1])])
        labelList.append(int(lineArr[2]))
    return dataList, labelList

def sigmoid(inX):
    return 1/(1+np.exp(-inX))

def simpleTest():
    filename= "AiLearning-master/data/5.Logistic/TestSet.txt"
    dataList, labelList = loadDataSet(filename)
    weights = gradDescent(dataList, labelList)
    plotBestFit(dataList, labelList, weights)



def gradDescent(dataList, classLabels):
    dataMatrix = np.mat(dataList) # 转换成矩阵形式
    learningRate = 0.001
    epoch = 500
    # m样本个数， n每个样本输入的特征数
    m, n = np.shape(dataMatrix)
    labelMat = np.mat(classLabels).transpose() #转换成[n, 1]
    weights = np.ones((n, 1))
    for epochIdx in range(epoch):
        yhat = sigmoid(dataMatrix * weights) #[m, n] * [n, 1] = [m, 1]
        error = yhat - labelMat # [m, 1]
        # w  = w - alpha * X * (yhat - y)
        weights = weights - learningRate*dataMatrix.transpose()*error
        #  + or -是根据yhat-y还是y-yhat来的
    return weights


def plotBestFit(dataArr, labelMat, weights):
    m = np.shape(dataArr)[0]
    xcord1 = []
    ycord1 = []
    xcord2 = []
    ycord2 = []
    for i in range(m):
        if int(labelMat[i])== 1:
            xcord1.append(dataArr[i][1])
            ycord1.append(dataArr[i][2])
        else:
            xcord2.append(dataArr[i][1])
            ycord2.append(dataArr[i][2])
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.scatter(xcord1, ycord1, s=30, c='red', marker='s')
    ax.scatter(xcord2, ycord2, s=30, c='green')
    x = np.arange(-3.0, 3.0, 0.1)
    """
    y的由来，卧槽，是不是没看懂？
    首先理论上是这个样子的。
    w0*x0+w1*x1+w2*x2=f(x)
    x0最开始就设置为1叻， x2就是我们画图的y值，而f(x)被我们磨合误差给算到w0,w1,w2身上去了
    所以： w0+w1*x+w2*y=0 => y = (-w0-w1*x)/w2   
    """
    y = np.array(((-weights[0] - weights[1] * x) / weights[2]).transpose()).flatten()
    ax.plot(x, y)
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.show()


simpleTest()
