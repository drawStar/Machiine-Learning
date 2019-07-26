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



# 打开测试集和训练集,并对数据进行格式化处理
def colicTest():
    '''
    Desc: 打开测试集和训练集，并对数据进行处理
    每个样本有21个特征，1个label
    Returns: errorRate -- 分类错误率
    '''
    frTrain = open('AiLearning-master/data/5.Logistic/horseColicTraining.txt')
    frTest = open('AiLearning-master/data/5.Logistic/horseColicTest.txt')
    trainingSet = []
    trainingLabels = []
    for line in frTrain.readlines():
        currentLine = line.strip().split('\t')
        lineArr = []
        for i in range(21):
            lineArr.append(float(currentLine[i]))
        trainingSet.append(lineArr)
        trainingLabels.append(float(currentLine[21]))

    trainWeights = stocGradAscent1(np.array(trainingSet), trainingLabels, 500)
    errorCount = 0
    numTestVec = 0.0  #测试样本数量
    # 读取 测试数据集 进行测试，计算分类错误的样本条数和最终的错误率
    for line in frTest.readlines():
        numTestVec += 1.0
        currLine = line.strip().split('\t')
        lineArr = []
        for i in range(21):
            lineArr.append(float(currLine[i]))
        if int(classifyVector(np.array(lineArr), trainWeights)) != int(currLine[21]):
            errorCount += 1
    errorRate = (float(errorCount) / numTestVec)
    print("the error rate of this test is: %f" % errorRate)
    return errorRate





def stocGradAscent1(dataMatrix, classLabels, epoch=150):
    m,n = np.shape(dataMatrix)
    weights = np.ones(n)
    for j in range(epoch):
        dataIndex = [x for x in range(m)]
        for i in range(m): #m是训练样本总数
            # i和j的不断增大，导致alpha的值不断减少，但是不为0
            alpha = 4 / (1.0 + j + i) + 0.0001  # alpha 会随着迭代不断减小，但永远不会减小到0，因为后边还有一个常数项0.0001
            # random.uniform(x, y) 方法将随机生成下一个实数，它在（x,y）范围内,x是这个范围内的最小值，y是这个范围内的最大值。
            randIndex = int(np.random.uniform(0, len(dataIndex)))
            # sum(dataMatrix[i]*weights)为了求 f(x)的值， f(x)=a1*x1+b2*x2+..+nn*xn
            # 因为是array相乘不是 mat相乘，所以多一步sum
            yhat = sigmoid(sum(dataMatrix[dataIndex[randIndex]]*weights))
            error = yhat - classLabels[dataIndex[randIndex]]
            weights = weights - alpha * error * dataMatrix[dataIndex[randIndex]]
            del (dataIndex[randIndex])
            """
            通过随机选取样本更新回归系数。这种方法将减少周期性的波动
            这种方法每次随机从列表中选出一个值，然后从列表中删掉该值（再进行下一次迭代）
            """
    return weights

# --------------------------------------------------------------------------------
# 从疝气病症预测病马的死亡率
# 分类函数，根据回归系数和特征向量来计算 Sigmoid的值
def classifyVector(inX, weights):
    '''
    Desc:         最终的分类函数，根据回归系数和特征向量来计算 Sigmoid 的值，大于0.5函数返回1，否则返回0
    Args: inX -- 特征向量，features
        weights -- 根据梯度下降/随机梯度下降 计算得到的回归系数
    Returns:
        如果 prob 计算大于 0.5 函数返回 1        否则返回 0
    '''
    prob = sigmoid(sum(inX * weights))
    if prob > 0.5:
        return 1.0
    else:
        return 0.0

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

# a=np.array([2,3,4])
# b=np.array([4,2,1])
# print(a*b)
colicTest()