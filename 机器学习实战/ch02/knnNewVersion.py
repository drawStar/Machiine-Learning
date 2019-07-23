import numpy as np
def file2matrix(filename):
    fr = open(filename) #特征1，特征2，特征3，类别
    numberOfLines = len(fr.readlines())  #文件行数
    returnMat = np.zeros((numberOfLines, 3))  # 存储data
    classLabelVector = [] # 存储label
    fr = open(filename)
    index = 0
    for line in fr.readlines():
        line = line.strip()
        listFromLine = line.split('\t')
        returnMat[index, :] = listFromLine[:-1]
        classLabelVector.append(int(listFromLine[-1]))
        index += 1
    return returnMat, classLabelVector

def autoNorm(dataSet):
    """
        归一化特征值，消除特征之间量级不同导致的影响
        公式：Y = (X-Xmin)/(Xmax-Xmin)
    """
    minValues = dataSet.min(axis=0)
    maxValues = dataSet.max(axis=0)
    ranges = maxValues - minValues # 极差
    normDataSet = dataSet - minValues
    normDataSet = normDataSet/ranges
    return normDataSet

def classify0(inX, trainDataSet, labels, k):
    """
    :param inX: 一个test样本
    :param dataSet: 全部train样本
    :param labels: t全部rain样本的labels
    :param k:
    :return:
    """
    distance = (inX-trainDataSet)**2
    sumDistance = distance.sum(axis=1)

    # 与全部train样本的欧氏距离
    euclideanDistance = sumDistance**0.5

    # 将距离从小到大排序,返回索引
    sortedDistIndicies = euclideanDistance.argsort()

    #选取前K个最短距离， 选取这K个中最多的分类类别
    classCount={}
    for i in range(k):
        votelLabel = labels[sortedDistIndicies[i]]
        classCount[votelLabel] = classCount.get(votelLabel, 0) + 1
    testLabel = sorted(classCount, key=lambda x: classCount[x])[-1]
    return testLabel


def datingClassTest():
        """
        :return: 错误数
        """
        trainRatio = 0.9 #训练样本所占比例
        filename = "AiLearning-master/data/2.KNN/datingTestSet2.txt"
        datingDataMat, datingLabels = file2matrix(filename)
        numberOfLines = datingDataMat.shape[0]
        numberOfTrain = int((trainRatio) * numberOfLines)  # 测试样本数量
        numberOfTest = numberOfLines - numberOfTrain  # 测试样本数量
        normDataSet = autoNorm(datingDataMat) # normlised dataset

        errorCount = 0
        k = 3

        for i in range(numberOfTest):
            # classifierResult = classify0(normDataSet[i, :], normDataSet[numberOfTest:numberOfLines, :], datingLabels[numberOfTest:numberOfLines], k)
            classifierResult = classify0(normDataSet[numberOfTrain + i, :], normDataSet[:numberOfTrain,:], datingLabels[:numberOfTrain], k)
            realLabel = datingLabels[numberOfTrain + i]
            # print("the classifier result=: %d, the real answer is: %d" % (classifierResult, realLabel))
            if (classifierResult != realLabel): errorCount += 1.0

        print("the total error rate is: %f" % (errorCount / float(numberOfTest)))
        # 最后结果错误率0.07

datingClassTest()