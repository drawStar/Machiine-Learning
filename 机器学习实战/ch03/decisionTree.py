import numpy as np
import math
import copy

def createDataSet():
    """
    feature0 年龄:青少年1，中年2，老年3
    feature1 层次：低1， 中2， 高3
    feature2 学生： 是1，否0
    feature3: 信用等级： 良好1， 一般0
    class：是否购买电脑： yes， no
    https://www.cnblogs.com/fengfenggirl/p/classsify_decision_tree.html    """
    dataSet = [[1, 3, 0, 0, 'no'],
               [1, 3, 0, 1, 'no'],
               [2, 3, 0, 0, 'yes'],
               [3, 2, 0, 0, 'yes'],
               [3, 1, 1, 0, 'yes'],
               [3, 1, 1, 1, 'no'],
               [2, 1, 1, 1, 'yes'],
               [1, 2, 0, 0, 'no'],
               [1, 1, 1, 0, 'yes'],
               [3, 2, 1, 0, 'yes'],
               [1, 2, 1, 1, 'yes'],
               [2, 2, 0, 1, 'yes'],
               [2, 3, 1, 0, 'yes'],
               [3, 2, 0, 1, 'no']]
    features = ['年龄', '层次', '是否学生', '信用等级']
    return dataSet, features
  #  labels = ['no surfacing', 'flippers']
  # return dataSet, labels

def calcShannonEnt(dataSet):
    # 计算信息熵
    numOfData = len(dataSet)
    labelCounts = {}
    for featureVector in dataSet:
        currentLabel = featureVector[-1]
        if currentLabel not in labelCounts.keys():
            labelCounts[currentLabel] = 0
        labelCounts[currentLabel] += 1

    shannonEntropy = 0.0

    for key in labelCounts:
        prob = float(labelCounts[key]/numOfData)
        shannonEntropy -= prob * math.log(prob, 2)
    return shannonEntropy


# 按给定特征值为value的来 划分数据集,返回去除特征值==value的样本后的数据集
def splitDataSet(dataSet, index, value):
    #index 为所选特征，即该特征所在的列
    restDataset = []  #剩下的特征
    for featureVector in dataSet:
        if featureVector[index] == value:
            #去除索引为index的特征, reducedFeatureVec保存该样本剩下的特征
            reducedFeatureVec = featureVector[:index] #[0,index)
            reducedFeatureVec.extend(featureVector[index+1:])
            restDataset.append(reducedFeatureVec) #注意区分extend和append
    return restDataset


def chooseBestFeatureToSplit(dataSet):
    # 求第一行有多少列的 Feature, -1是因为最后一列是label列
    numFeatures = len(dataSet[0]) - 1
    # 数据集的原始信息熵
    baseEntropy = calcShannonEnt(dataSet)
    # 最优的信息增益值, 和最优的Featurn编号
    bestInfoGain, bestFeature = 0.0, -1
    for i in range(numFeatures): # 对每一个特征
        featList = [example[i] for example in dataSet] #获得这一列的features
        # 获取剔重后的集合，使用set对list数据进行去重，获得这个特征的所有value
        uniqueVals = set(featList)
        newEntropy = 0.0            # 创建一个临时的信息熵
        # 遍历某一列的value集合，计算该列的信息熵
        # 遍历当前特征中的所有唯一属性值，对每个唯一属性值划分一次数据集，计算数据集的新熵值，并对所有唯一特征值得到的熵求和。
        for value in uniqueVals:
            subDataSet = splitDataSet(dataSet, i, value)
            prob = float(len(subDataSet)/len(dataSet))
            newEntropy += prob * calcShannonEnt(subDataSet)

        infoGain = baseEntropy - newEntropy #计算信息增益
        if (infoGain > bestInfoGain):
            bestInfoGain = infoGain
            bestFeature = i
    # print('infoGain=', infoGain, 'bestFeature=', bestFeature)
    return bestFeature

def majorityCnt(classList):
    classCount = {}
    for item in classList:
        classCount[item] = classCount.get(item, 0) + 1
    sortedClassList = sorted(classCount, key=lambda x: classCount[x] )

    return sortedClassList[-1]

def createTree(dataSet, featureNames):
    # ID3 算法
    classList = [example[-1] for example in dataSet]
    #第一个停止条件,  给定结点的所有样本的类标签相同，则直接返回该类标签。
    if classList.count(classList[0]) == len(classList):
        return classList[0]
    # 第二个停止条件：使用完了所有特征，仍然不能将数据集划分成仅包含唯一类别的分组，这样情况下，使用多数表决，用样本中的多数所处类标记
    if len(dataSet[0]) == 1:
        return majorityCnt(classList)

    #选择当前数据集下最优的特征
    bestFeatureID = chooseBestFeatureToSplit(dataSet)
    bestFeatureName = featureNames[bestFeatureID]
    bestFeatValues = [example[bestFeatureID] for example in dataSet]

    featureNames.remove(bestFeatureName)  # 去除最优的featureName
    myTree = {bestFeatureName:{}} #用dict存储
    uniqueVals = set(bestFeatValues)
    for value in uniqueVals:
        # 求出剩余的标签label
        subFeatureNames = featureNames[:]  # TODO:为什么不能放在for循环外面?
        # for循环的第一个value可能会改变subFeatureNAmes 的值
        subDataSet = splitDataSet(dataSet, bestFeatureID, value) #按最优特征划分数据集
        myTree[bestFeatureName][value] = createTree(subDataSet, subFeatureNames)

    return myTree

def fishTest():
    # 1.创建数据和结果标签
    myData, featureNames = createDataSet()
    # print(featureNames)
    featureNamesCopy = featureNames.copy()
    myTree = createTree(myData, featureNames)
    # print(featureNames)  #featureNames的值改变了

    testSanmple = [3, 2, 0, 1]
    classLabel=classify(myTree, featureNamesCopy, testSanmple)
    print(classLabel)

def classify(inputTree, featureNames, testSanmple):
    rootFeatureName = list(inputTree.keys())[0] #决策树根节点名称
    # print(inputTree)
    rootFeatureIdx = featureNames.index(rootFeatureName)
    secondDict = inputTree[rootFeatureName]
    # print(secondDict)
    for key in secondDict: #key是feature的value
        if key == testSanmple[rootFeatureIdx]:
            if type(secondDict[key]).__name__ == 'dict':
                classLabel = classify(secondDict[key], featureNames, testSanmple)
            else:
                classLabel = secondDict[key]
    return classLabel







fishTest()