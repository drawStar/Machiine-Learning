# 项目案例1: 屏蔽社区留言板的侮辱性言论

import numpy as np


def loadDataSet():
    """
    创建数据集,每行表示一个文档，每个文档属于一个类别（01表示，1表示侮辱）
    :return: 单词列表postingList, 所属类别classVec
    """
    postingList = [['my', 'dog', 'has', 'flea', 'problems', 'help', 'please'], #[0,0,1,1,1......]
                   ['maybe', 'not', 'take', 'him', 'to', 'dog', 'park', 'stupid'],
                   ['my', 'dalmation', 'is', 'so', 'cute', 'I', 'love', 'him'],
                   ['stop', 'posting', 'stupid', 'worthless', 'garbage'],
                   ['mr', 'licks', 'ate', 'my', 'steak', 'how', 'to', 'stop', 'him'],
                   ['quit', 'buying', 'worthless', 'dog', 'food', 'stupid']]
    classVec = [0, 1, 0, 1, 0, 1]  # 1 is abusive, 0 not
    return postingList, classVec


def createVocabList(dataSet):
    """
    获取所有单词的集合
    :param dataSet: 数据集
    :return: 所有单词的集合(即不含重复元素的单词列表)
    """
    vocabSet = set([])  # create empty set
    for document in dataSet:
        # 操作符 | 用于求两个集合的并集
        vocabSet = vocabSet | set(document)  # union of the two sets
    return list(vocabSet)


def setOfWords2Vec(vocabList, inputSet):
    """
    遍历查看该单词是否出现，出现该单词则将该单词置1
    :param vocabList: 所有单词集合列表
    :param inputSet: 输入数据集
    :return: 匹配列表[0,1,0,1...]，其中 1与0 表示词汇表中的单词是否出现在输入的数据集中
    """
    # 创建一个和词汇表等长的向量，并将其元素都设置为0
    returnVec = [0] * len(vocabList)# [0,0......]
    # 遍历文档中的所有单词，如果出现了词汇表中的单词，则将输出的文档向量中的对应值设为1
    for word in inputSet:
        if word in vocabList:
            returnVec[vocabList.index(word)] = 1
    return returnVec

def trainNB0(trainMatrix, trainCategory):
    # 给定一文档，判断属于哪一类，即计算P（ci|w）属于哪个ci时最大
    # P(ci|w)=P(ci)*P(w|ci) / P(w), 对于每个P(ci|w)来说，P（w）相同，不计算了。
    # P(ci) 属于ci的文档出现的概率=count（属于ci的文档）/count（总文档数）
    # P（w|ci）使用 p(w0 | ci)*p(w1 | ci)*p(w2 | ci)...p(wn | ci)
    #P（w0|ci）wo在ci的文档中出现的次数/ w0在所有文档出现的总次数 错
    #P（w0|ci）wo在ci的文档中出现的次数/ ci的文档中出现的单词总数 对


    #trainMatrix [[1,0,1,1,0,...],...] 10表示这个单词是否在V中出现过
    numTrainDocs = len(trainMatrix) #训练总文档个数
    numWords = len(trainMatrix[0]) #词汇表大小

    # 侮辱性文档出现的概率=count（侮辱性文档）/count（总文档数）
    pAbusive = sum(trainCategory) / numTrainDocs


    p1Num = np.ones(numWords) # [1,1,1,.....] 记录每个单词出现的次数
    p0Num = np.ones(numWords) #不能初始化为0，因为会使概率为0

    numAbusiveWords = 2.0 #2.0根据样本/实际调查结果调整分母的值（2主要是避免分母为0，值可以调整）
    numNoAbusiveWords = 2.0

    for i in range(numTrainDocs):
        if trainCategory[i] == 1: #若是侮辱性文档
            # 计算所有侮辱性文档中出现的单词总数
            numAbusiveWords += sum(trainMatrix[i])
            p1Num += trainMatrix[i]  # [0,1,1,....] + [0,1,1,....]->[0,2,2,...]
        else:
            numNoAbusiveWords += sum(trainMatrix[i])
            p0Num += trainMatrix[i]

    #为防止概率太小，使用log改进 log（P(ci)*P(w|ci））=log（P（ci））+log(P(w|ci))
    # = log（P（ci））+ log(P(w0|ci))+log(P(w1|ci))+...log(P(wn|ci))
    # 向量 每个单词在侮辱性文档出现的概率P(wi|c1)
    p1Vec = np.log(p1Num / numAbusiveWords)
    # 向量 每个单词在正常文档出现的概率P(wi|c0)
    p0Vec = np.log(p0Num / numNoAbusiveWords)

    return p0Vec, p1Vec, pAbusive

def classifyNB(testVec, p0Vec, p1Vec, pAbusive):
    """
       使用算法：  # 将乘法转换为加法
           乘法：P(C|F1F2...Fn) = P(F1F2...Fn|C)P(C)/P(F1F2...Fn)
           加法：P(F1|C)*P(F2|C)....P(Fn|C)P(C) -> log(P(F1|C))+log(P(F2|C))+....+log(P(Fn|C))+log(P(C))
       :param vec2Classify: 待测数据[0,1,1,1,1...]，即要分类的向量
       :param p0Vec: 类别0，即正常文档的[log(P(F1|C0)),log(P(F2|C0)),log(P(F3|C0)),log(P(F4|C0)),log(P(F5|C0))....]列表
       :param p1Vec: 类别1，即侮辱性文档的[log(P(F1|C1)),log(P(F2|C1)),log(P(F3|C1)),log(P(F4|C1)),log(P(F5|C1))....]列表
       :param pClass1: 类别1，侮辱性文件的出现概率
       :return: 类别1 or 0
    """
    # 计算公式  log(P(F1|C))+log(P(F2|C))+....+log(P(Fn|C))+log(P(C))
    # 计算P(ci|w)=P(ci)*P(w|ci) / P(w), 对于每个P(ci|w)来说，P（w）相同，不计算了。


    p1 = sum(testVec * p1Vec) + np.log(pAbusive)  # P(w|c1) * P(c1) ，即贝叶斯准则的分子
    p0 = sum(testVec * p0Vec) + np.log(1.0 - pAbusive)  # P(w|c0) * P(c0) ，即贝叶斯准则的分子·
    if p1 > p0:
        return 1
    else:
        return 0


def simpleTest():
    """
    测试朴素贝叶斯算法
    """
    # 1. 加载数据集
    listOfPosts, listClasses = loadDataSet()
    # 2. 创建单词集合
    myVocabList = createVocabList(listOfPosts)
    # 3. 计算单词是否出现并创建数据矩阵
    trainMat = []
    for doc in listOfPosts:
        # 返回m*len(myVocabList)的矩阵， 记录的都是0，1信息
        trainMat.append(setOfWords2Vec(myVocabList, doc))

    # 4. 训练数据
    p0V, p1V, pAb = trainNB0(np.array(trainMat), np.array(listClasses))
    # 5. 测试数据
    testDoc = ['love', 'my', 'dalmation']
    testVec = np.array(setOfWords2Vec(myVocabList, testDoc))
    print(testDoc, 'classified as: ', classifyNB(testVec, p0V, p1V, pAb))

    testDoc = ['stupid', 'garbage']
    testVec = np.array(setOfWords2Vec(myVocabList, testDoc))
    print(testDoc, 'classified as: ', classifyNB(testVec, p0V, p1V, pAb))



simpleTest()
