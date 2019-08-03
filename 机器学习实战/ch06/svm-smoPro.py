
import numpy as np
import matplotlib.pyplot as plt
import random

class optStruct:
    # dataMatIn - 数据矩阵
    # classLabels - 数据标签
    # C - 松弛变量
    # toler - 容错率
    def __init__(self, dataMatIn, classLabels, C, toler):
        self.X = dataMatIn
        self.labelMat = classLabels
        self.C = C
        self.tol = toler
        self.m = np.shape(dataMatIn)[0] #样本数
        self.alphas = np.mat(np.zeros((self.m, 1)))
        self.b = 0

        # 根据矩阵行数初始化虎误差缓存，第一列为是否有效的标志位，第二列为实际的误差E的值。
        self.eCache = np.mat(np.zeros((self.m, 2))) #TODO:
        #


def loadDataSet(filename):
    dataList = []
    labelList = []
    fr = open(filename)
    for line in fr.readlines():
        lineArr = line.strip().split('\t')
        dataList.append([float(lineArr[0]), float(lineArr[1])])
        labelList.append(float(lineArr[2]))
    return dataList, labelList

def calcEk(oS, k): #计算误差
    fXk = float(np.multiply(oS.alphas, oS.labelMat).T * (oS.X * oS.X[k, :].T) + oS.b)
    Ek = fXk - float(oS.labelMat[k])

    return Ek


def clipAlpha(aj,H,L):
    """
    修剪alpha_j
    Parameters:
        aj - alpha_j的值
        H - alpha上限
        L - alpha下限
    Returns:
        aj - 修剪后的alpah_j的值
    """
    if aj > H:
        aj = H
    if L > aj:
        aj = L
    return aj

def updateEk(oS,k):
    Ek=calcEk(oS,k)
    oS.eCache[k]=[1, Ek]


def innerL(i, oS):
    """
    优化的SMO算法， 内部循环代码
    Parameters：
        i - 标号为i的数据的索引值
        oS - 数据结构
    Returns:
        1 - 有任意一对alpha值发生变化
        0 - 没有任意一对alpha值发生变化或变化太小
    """
    # 步骤1 计算误差Ei
    Ei = calcEk(oS, i)
    if ((oS.labelMat[i] * Ei < -oS.tol) and (oS.alphas[i] < oS.C)) or \
            ((oS.labelMat[i] * Ei > oS.tol) and (oS.alphas[i] > 0)):
        #内循环选择alphaj
        j, Ej = selectJ(i, oS, Ei)
        alphaIold = oS.alphas[i].copy()
        alphaJold = oS.alphas[j].copy()

        # 步骤2：计算上下界L和H
        if (oS.labelMat[i] != oS.labelMat[j]):
            L = max(0, oS.alphas[j] - oS.alphas[i])
            H = min(oS.C, oS.C + oS.alphas[j] - oS.alphas[i])
        else:
            L = max(0, oS.alphas[j] + oS.alphas[i] - oS.C)
            H = min(oS.C, oS.alphas[j] + oS.alphas[i])
        if L == H:  # TODO:
            # 如果L==H，就不做任何改变，直接执行continue语句
            return 0

        # 计算eta
        eta = oS.X[i, :] * oS.X[i, :].transpose() + oS.X[j, :] * oS.X[j, :].transpose() - 2.0 * oS.X[i, :] * oS.X[j,
                                                                                                             :].transpose()
        if eta == 0:
            return 0  # 第二个跳出条件(因为eta=0不好处理，且出现情况较少，因此这里咱不处理，直接跳出)

        # 步骤4：更新alpha_j
        oS.alphas[j] += oS.labelMat[j] * (Ei - Ej) / eta
        # 步骤5：修剪alpha_j
        oS.alphas[j] = clipAlpha(oS.alphas[j], H, L)


        # 检验alphaj与alphaJold是否有足够大的改变，若改变不够大，说明与alpha旧值没有什么差异，跳出本次内循环
        if abs(oS.alphas[j] - alphaJold) < 0.00001:
            # print("j not moving enough")
            return 0  # 第三个跳出条件
        # 更新Ej至误差缓存
        updateEk(oS, j)
        # 约束条件让我们可以根据alphaJ求出alphaI
        oS.alphas[i] += oS.labelMat[i] * oS.labelMat[j] * (alphaJold - oS.alphas[j])
        updateEk(oS, i)  # 更新Ei值

        # 更新b值,根据alpha是否在0～C决定更新的b值
        b1 = -Ei - oS.labelMat[i] * (oS.alphas[i] - alphaIold) * oS.X[i, :] * oS.X[i, :].transpose() \
             - oS.labelMat[j] * (oS.alphas[j] - alphaJold) * oS.X[j, :] * oS.X[i, :].transpose() + oS.b

        b2 = -Ej - oS.labelMat[i] * (oS.alphas[i] - alphaIold) * oS.X[i, :] * oS.X[j, :].transpose() \
             - oS.labelMat[j] * (oS.alphas[j] - alphaJold) * oS.X[j, :] * oS.X[j, :].transpose() + oS.b
        # 若ai或aj在(0,C)之间，则取b=bi或b=bj，若ai aj都不在(0,C)之间，取均值
        if oS.alphas[i] > 0 and oS.alphas[i] < oS.C:
            oS.b = b1
        elif oS.alphas[j] > 0 and oS.alphas[j] < oS.C:
            oS.b = b2
        else:
            oS.b = (b1 + b2) / 2.0

        updateEk(oS, i)
        updateEk(oS, j)

        return 1  # 若执行到这里都没有return0跳出，说明已经完成了一个alpha对的更新，返回一个1

    else:
        return 0



def selectJ(i, oS, Ei):
    #选择|Ej-Ei|最大的alphaj
    maxK = -1 #|Ei-Ej|最大时的索引j
    maxDeltaE = 0
    Ej = 0
    oS.eCache[i] = [1, Ei]  # 根据Ei更新误差缓存
    validEcacheList = np.nonzero(oS.eCache[:, 0])[0]  # 返回误差不为0的数据的索引值
    #todo： why不从全部alpha中选？
    if (len(validEcacheList)) > 1:
        for k in validEcacheList:
            if k == i:
                continue #|Ek-Ei|=0,肯定不是最大值，所以跳过
            Ek = calcEk(oS, k) # todo :Ek和eCache[k][0,1]区别,why两个值会不同？
            #why要重新计算一次 b发生改变后而没有更新Ek
            print("Ek", Ek)
            print(oS.eCache[k][0,1])
            if(Ek == oS.eCache[k][0,1]):
                print("y")
            else:
                print("n")

            deltaE = abs(Ei-Ek)
            if(deltaE > maxDeltaE):
                maxDeltaE = deltaE
                Ej = Ek
                maxK = k
        return maxK, Ej

    else: #没有不为0的误差
        j = selectJrand(i, oS.m)
        Ej = calcEk(oS, j)
    return j, Ej


def selectJrand(i, m):
    #返回一个不为i的随机数j，在0~m之间的整数值
    j = i
    while j == i:
        j = int(random.uniform(0, m))
    return j

def smoP(dataMatIn, classLabels, C, toler, maxIter):
    """
    完整的线性SMO算法
    Parameters：
        dataMatIn - 数据矩阵
        classLabels - 数据标签
        C - 松弛变量
        toler - 容错率
        maxIter - 最大迭代次数
    Returns:
        oS.b - SMO算法计算的b
        oS.alphas - SMO算法计算的alphas
    """

    oS = optStruct(np.mat(dataMatIn), np.mat(classLabels).transpose(), C, toler)  # 初始化数据结构
    iter = 0  # 初始化当前迭代次数
    entireSet = True
    alphaPairsChanged = 0
    while (iter < maxIter) and ((alphaPairsChanged > 0) or (entireSet)):
        # 遍历整个数据集都alpha也没有更新  或者 超过最大迭代次数, 则退出循环
        alphaPairsChanged = 0
        if entireSet:
            # 遍历整个数据集，选择第一个参数alphai
            for i in range(oS.m):
                alphaPairsChanged += innerL(i, oS) #todo: +=? +=没啥用，只是记录是否变化而已
            iter+=1 #这里的迭代次数只要完成一次循环遍历就+1，不论该次循环遍历是否修改了alpha对

        else:#遍历非边界alpha
            boundIs = np.nonzero((oS.alphas.A > 0) * (oS.alphas.A < oS.C))[0]  # 选择0<alpha<C的样本点的索引值(即边界点)
            for i in boundIs:
                alphaPairsChanged += innerL(i, oS)
            iter += 1

        # 控制遍历往返于全集遍历和边界遍历
        if entireSet:
            entireSet = False  # 若本轮是全集遍历，则下一轮进入边界遍历(下一轮while条件中的entire是False)
        elif alphaPairsChanged == 0:# 若本轮是非边界遍历，且本轮遍历未修改任何alpha对，则下一轮进入全集遍历
            entireSet = True

    return oS.b, oS.alphas


def showClassifer(dataMat, w, b):
    #绘制样本点
    posData = []                                  #正样本
    negData = []                                 #负样本
    for i in range(len(dataMat)):
        if classLabels[i] > 0:
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

def getW(alphas,dataArr,classLabels):
    """
    计算w
    Parameters:
        dataArr - 数据矩阵
        classLabels - 数据标签
        alphas - alphas值
    Returns:
        w - 计算得到的w
    """
    X = np.mat(dataArr); labelMat = np.mat(classLabels).transpose()
    m,n = np.shape(X)
    w = np.zeros((n,1))
    for i in range(m):
        w += np.multiply(alphas[i]*labelMat[i],X[i,:].T)
    return w

if __name__ == '__main__':
    dataArr, classLabels = loadDataSet('AiLearning-master/data/6.SVM/testSet.txt')
    b, alphas = smoP(dataArr, classLabels, 0.6, 0.001, 70)
    w = getW(alphas,dataArr, classLabels)
    showClassifer(dataArr, w, b)