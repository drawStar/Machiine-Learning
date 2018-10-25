"""
datingTestSet 第一列 每年获得的飞行常客里程数， 第二列 玩游戏所耗时间百分比，第三列 每周消费的冰淇淋公升数

"""

import numpy as np
import math
import operator
import os
import matplotlib
import matplotlib.pyplot as plt

def createDataSet():
    group = np.array([[1.0, 1.1], [1.0, 1.0], [0, 0], [0, 0.1]])
    labels = ['A', 'A', 'B', 'B']
    return group, labels

def classify0(input, dataSet, labels, k):
    #input size为[1,2]，labels为dataSet数据的每个样本的标签

    #计算欧式距离
    dataSetSize = dataSet.shape[0]
    sqout = (np.tile(input, (dataSetSize, 1))-dataSet)**2
    #input行扩为datasetSize的大小，列不变
    distance = (sqout.sum(axis=1))**0.5

    #取距离最小的k个点
    classCount = {}
    index = distance.argsort()
    # 取距离最小的k个点在dataSet中的索引
    for i in range(k):
        classCount[labels[index[i]]] = classCount.get(labels[index[i]], 0) + 1
        #字典的get（key, default=None), 返回指定键的值，如值不在字典中返回默认值

    #将classCount进行排序
    soretedclassCount = sorted(classCount.items(), key = lambda x:x[1], reverse = True)
    return soretedclassCount[0][0]


def file2matrix(filename):
    fr = open(filename)
    arrayOfLines = fr.readlines()
    numOfLines = len(arrayOfLines) # 获取文件行数
    returnMat = np.zeros((numOfLines, 3))
    classLabelVector = []
    index = 0
    for line in (arrayOfLines):
        line = line.strip() # 去掉首尾空格
        listFromLine = line.split('\t')
        returnMat[index, :] = listFromLine[0:3]
        if listFromLine[-1] == 'largeDoses':
            classLabelVector.append(2)
        elif listFromLine[-1] == 'smallDoses':
            classLabelVector.append(1)
        elif listFromLine[-1] == 'didntLike':
            classLabelVector.append(0)
        index += 1
    return returnMat, classLabelVector
def autoNorm(dataSet):
    # 归一化特征值， newvalue = (oldvalue-min)/(max-min)
    minValues = dataSet.min(axis=0) #按列选择最小值
    maxValues = dataSet.max(axis=0)
    ranges = maxValues - minValues
    normDataSet = np.zeros(dataSet.shape)
    # 因为minValues和maxValues的shape为[1,3]，需要tile
    normDataSet = dataSet - np.tile(minValues, (dataSet.shape[0], 1))
    normDataSet = normDataSet/np.tile(ranges, (dataSet.shape[0], 1))
    return normDataSet, ranges, minValues
def datingClassTest():
    #测试
    hoRatio = 0.10 # 取10%用来测试
    datingDataMat, datingLabels = file2matrix('datingTestSet.txt')
    normMat, ranges, minVals = autoNorm(datingDataMat)
    m = normMat.shape[0]
    numTestVecs = int(m*hoRatio) #测试样本数量
    errorCount = 0.0
    for i in range(numTestVecs):
        predictResult = classify0(normMat[i, :], normMat[numTestVecs:m, :], datingLabels[numTestVecs:m], k=3)
        #前10%test，后90%train
        # print("the prediction result with: %d, the real answer is: %d" % (predictResult, datingLabels[i]))
        if (predictResult != datingLabels[i]): errorCount += 1.0
    print("the total error rate is: %f" % (errorCount / float(numTestVecs)))
    print(errorCount)

def classifyPerson():
    # 手动输入三个特征值，输出与海伦配对的可能性
    resultList = ['not at all', 'in small does', 'in large does']
    feature1 = float(input("每周玩游戏时间所占百分比"))
    feature2 = float(input("每年获得的飞行常客里程数"))
    feature3 = float(input("每周消费的冰淇淋公升数"))
    datingDataMat, datingLabels = file2matrix('datingTestSet.txt')
    normMat, ranges, minVals = autoNorm(datingDataMat)
    inArr = np.array([feature1, feature2, feature3])
    classResult = classify0((inArr - minVals)/ranges, normMat, datingLabels, 3)
    print(resultList[classResult])


def img2vector(filename):
    # 将 32*32 的二进制图像(实际是32*32的txt文档)转为1*1024的向量
    returnVector = np.zeros((1, 1024))
    fr = open(filename)
    for i in range(32):
        line = fr.readline()
        for j in range(32):
            returnVector[0, 32*i+j] = int(line[j])
    return returnVector

def handwritingClassTest():
    hwLabels = []
    # 获取trainging文件列表
    trainingFileList = os.listdir('D:/buaa/hang/ml实战/MLiA_SourceCode/machinelearninginaction/Ch02/digits/trainingDigits')
    m = len(trainingFileList)# 训练样本个数
    traingMat = np.zeros((m, 1024))
    for i in range(m):
        # fileName = trainingFileList[i].split('_')[0]
        fileName = trainingFileList[i]  # 获取每个文件的名字0_0.txt
        classNum = int(fileName.split('_')[0])  # 获取类名0-9
        hwLabels.append(classNum)
        traingMat[i,:] = img2vector('D:/buaa/hang/ml实战/MLiA_SourceCode/machinelearninginaction/Ch02/digits/trainingDigits/%s' % fileName)

    testFileList = os.listdir('D:/buaa/hang/ml实战/MLiA_SourceCode/machinelearninginaction/Ch02/digits/testDigits')
    mTest = len(testFileList)# 训练样本个数
    errorCount = 0.0
    # testMat = np.zeros((mTest, 1024))
    for i in range(mTest):
        fileName = testFileList[i]  # 获取每个文件的名字0_0.txt

        classNum = int(fileName.split('_')[0])  # 获取类名0-9
        # hwLabels.append(classNum)
        testVec = img2vector('D:/buaa/hang/ml实战/MLiA_SourceCode/machinelearninginaction/Ch02/digits/testDigits/%s' % fileName)
        classifierResult = classify0(testVec, traingMat, hwLabels, 3)
        if (classifierResult != classNum):
            errorCount += 1
        print("errorCount=",errorCount,"i=%d /946" % i)
    print("the total error rate is: %f" % (errorCount / float(mTest)))

# group, labels = createDataSet()
# print(group,labels)
# print(classify0([0, 0], group, labels, 3))
# datingClassTest()
# classifyPerson()

# mm = img2vector('D:/buaa/hang/ml实战/MLiA_SourceCode/machinelearninginaction/Ch02/digits/testDigits/0_13.txt')
# print(mm.shape)

handwritingClassTest()


