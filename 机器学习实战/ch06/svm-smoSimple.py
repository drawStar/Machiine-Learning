import numpy as np
import random
import matplotlib.pyplot as plt

def loadDataSet(filename):
    dataList = []
    labelList = []
    fr = open(filename)
    for line in fr.readlines():
        lineArr = line.strip().split('\t')
        dataList.append([float(lineArr[0]), float(lineArr[1])])
        labelList.append(float(lineArr[2]))
    return dataList, labelList

def showDataSet(dataMat, labelMat):
    posData = []
    negData = []
    for i in range(len(dataMat)):
        if labelMat[i] > 0:
            posData.append(dataMat[i])
        else:
            negData.append(dataMat[i])
    posData = np.array(posData).transpose() # 转置是为了画图方便
    negData = np.array(negData).transpose()

    plt.scatter(posData[0], posData[1])  #正样本数据点
    plt.scatter(negData[0], negData[1])  #负样本数据点
    plt.show()

def selectJrand(i, m):
    #返回一个不为i的随机数j，在0~m之间的整数值
    j = i
    while j == i:
        j = int(random.uniform(0, m))
    return j


def clipAlpha(aj, H, L):
    # 调整aj的值，使aj处于 L <= aj <= H
    if aj > H:
        aj = H
    if L > aj:
        aj = L
    return aj

def smoSimple(dataMatIn, classLabels, C, toler, epoch):
    """
    :param C:松弛变量(常量值)，允许有些数据点可以处于分隔面的错误一侧。
            控制最大化间隔和保证大部分的函数间隔小于1.0这两个目标的权重。
            可以通过调节该参数达到不同的结果。
    :param toler:容错率（是指在某个体系中能减小一些因素或选择对某个系统产生不稳定的概率。）
            如果误差的绝对值大于toler，进行优化
    :return:
        b       模型的常量值
        alphas  拉格朗日乘子
    """
    # 转换为numpy的mat存储
    dataMatrix = np.mat(dataMatIn) #[m,2]
    labelMat = np.mat(classLabels).transpose() #[m,1]

    # 初始化b参数，统计dataMatrix的维度
    b = 0
    m, n = np.shape(dataMatrix)  #m 样本数 n特征数

    # 初始化alpha参数，设为0
    alphas = np.mat(np.zeros((m, 1)))

    epochIndex = 0     # 没有任何alpha改变的情况下遍历数据的次数
    while (epochIndex < epoch):
        alphaPairsChanged = 0  #记录alpha优化的次数

        for i in range(m):
            # 步骤1 计算Ei
            temp = np.multiply(alphas, labelMat) #multiply([m ,1],[m,1]) = [m, 1] # multiply 对应元素相乘，不求和
            temp2 = dataMatrix * dataMatrix[i, :].T #[m, 2] * [1, 2].T = [m, 1]
            fXi = float(temp.T * temp2) + b #float([1,1])
            #  我们预测的类别 y = w^Tx[i]+b; 其中因为 w = Σ(1~n) alpha[n]*lable[n]*x[n]

            Ei = fXi - labelMat[i]  #预测结果与真实结果比对，计算误差Ei=yhat - y

            #若不满足kkt条件，进行优化
            """由KKT条件可以推出
            yi*f(i) >= 1 and alpha = 0 (outside the boundary)
            yi*f(i) == 1 and 0<alpha< C (on the boundary)
            yi*f(i) <= 1 and alpha = C (between the boundary)
            https://www.twblogs.net/a/5b7f71952b717767c6afa718
            """
            if((labelMat[i]*Ei < -toler and alphas[i] < C) or (labelMat[i]*Ei>toler and alphas[i] > 0)):
                # 随机选择另一个与alpha_i成对优化的alpha_j
                j = selectJrand(i, m)
                # 计算误差Ej
                fXj = float(np.multiply(alphas, labelMat).T * (dataMatrix * dataMatrix[j, :].T)) + b
                Ej = fXj - float(labelMat[j])
                alphaIold = alphas[i].copy()#保存更新前的aplpha值，使用深拷贝
                alphaJold = alphas[j].copy()
                # 步骤2：计算上下界L和H
                if (labelMat[i] != labelMat[j]): #yi不同号时
                    L = max(0, alphas[j] - alphas[i])
                    H = min(C, C + alphas[j] - alphas[i])
                else:
                    L = max(0, alphas[j] + alphas[i] - C)
                    H = min(C, alphas[j] + alphas[i])
                if L == H:  #TODO:
                    #如果L==H，就不做任何改变，直接执行continue语句
                    continue
                # 步骤3：计算分母 eta
                # eta = 2.0*dataMatrix[i,:]*dataMatrix[j,:].T - dataMatrix[i,:]*dataMatrix[i,:].T - dataMatrix[j,:]*dataMatrix[j,:].T
                eta = dataMatrix[i,:]*dataMatrix[i,:].T + dataMatrix[j,:]*dataMatrix[j,:].T - 2.0*dataMatrix[i,:]*dataMatrix[j,:].T
                if eta == 0:
                    continue #TODO：why大于0也要退出
                # 步骤4：更新alpha_j
                # alphas[j] -= labelMat[j] * (Ei - Ej) / eta
                alphas[j] += labelMat[j] * (Ei - Ej) / eta

                # 步骤5：修剪alpha_j
                alphas[j] = clipAlpha(alphas[j], H, L)
                if (abs(alphas[j] - alphaJold) < 0.00001):
                    # print("alpha_j变化太小")
                    continue

                # 步骤6：更新alpha_i
                alphas[i] += labelMat[j] * labelMat[i] * (alphaJold - alphas[j])
                # 步骤7：更新b_1和b_2
                b1 = - Ei + (alphaIold - alphas[i])*labelMat[i] * dataMatrix[i, :] * dataMatrix[i, :].T  +\
                     (alphaJold-alphas[j])*labelMat[j]* dataMatrix[i, :] * dataMatrix[j, :].T +b

                b2 = - Ej + (alphaIold- alphas[i])*labelMat[i] * dataMatrix[i, :] * dataMatrix[j, :].T + \
                     (alphaJold - alphas[j] ) *labelMat[j] * dataMatrix[j, :] * dataMatrix[j, :].T + b
                # 步骤8：根据b_1和b_2更新b
                if (0 < alphas[i]) and (C > alphas[i]):
                    b = b1
                elif (0 < alphas[j]) and (C > alphas[j]):
                    b = b2
                else:
                    b = (b1 + b2) / 2.0
                alphaPairsChanged += 1  # 统计优化次数
        # 在for循环外，检查alpha值是否做了更新，如果在更新则将iter设为0后继续运行程序
        # 知道更新完毕后，iter次循环无变化，才推出循环。
        if (alphaPairsChanged == 0):
            epochIndex += 1
        else:
            epochIndex = 0
    return b, alphas

def get_w(dataList, labelList, alphas):
    # # w = Σ[1~n] ai * yi * xi

    X = np.mat(dataList)
    labelMat = np.mat(labelList).transpose()
    m, n = np.shape(X) # m 样本数 n特征数
    w = np.zeros((n, 1))
    for i in range(m):
        w += np.multiply(alphas[i] * labelMat[i], X[i, :].T)
    return w.tolist()

def showClassifer(dataMat, w, b):
    #绘制样本点
    posData = []                                  #正样本
    negData = []                                 #负样本
    for i in range(len(dataMat)):
        if labelList[i] > 0:
            posData.append(dataMat[i])
        else:
            negData.append(dataMat[i])
    posData_np = np.array(posData)              #转换为numpy矩阵
    negData_np = np.array(negData)            #转换为numpy矩阵
    plt.scatter(np.transpose(posData_np)[0], np.transpose(posData_np)[1], s=30, alpha=0.7)   #正样本散点图
    plt.scatter(np.transpose(negData_np)[0], np.transpose(negData_np)[1], s=30, alpha=0.7) #负样本散点图
    #绘制直线
    x1 = max(dataMat)[0]
    x2 = min(dataMat)[0]
    b = float(b)

    # 根据决策边界 x.w + b = 0 得到，其式子展开为w1.x1 + w2.x2 + b = 0, x2就是y值
    # x2 = (-b-w1x1) / w2
    w = np.squeeze(w)  #squeeze 去掉冗余的1维度
    w1=float(w[0])
    w2=float(w[1])


    y1 = (-b-w1*x1)/w2
    y2 = (-b-w1*x2)/w2
    print(y1,y2)

    plt.plot([x1, x2], [y1, y2]) # 画直线
    #找出支持向量点 alpha大于0小于C的点为支持向量
    for i, alpha in enumerate(alphas):
        if abs(alpha) > 0:
            x, y = dataMat[i]
            plt.scatter([x], [y], s=150, c='none', alpha=0.7, linewidth=1.5, edgecolor='red')
    plt.show()

if __name__ == "__main__":
    # 获取特征和目标变量
    dataList, labelList = loadDataSet('AiLearning-master/data/6.SVM/testSet.txt')
    # print(dataArr[1])
    # showDataSet(dataMat, labelMat)
    b, alphas = smoSimple(dataList, labelList, 0.6, 0.001, 40)
    w = get_w(dataList, labelList, alphas)
    showClassifer(dataList, w, b)
