pytorch实现姓氏分类
根据姓名推测姓名所属国籍，多分类

分别用MLP(多层感知机)和CNN实现

#### Abstract

embedding使用的one-hot矩阵表示，大小为[Vocabulary_size，maxSeqLength]
Model分别是MLP和CNN
CrossEntropyLoss 实现 多分类


主要涉及三个辅助类：
Vocabulary ： mapping text tokens to intergers and mapping the class labels to integers   .
ReviewDataset : 包含DataFrame格式的数据集，和ReviewVectorizer类的实例化vectorizer
ReviewVectorizer： 将word转为向量，包括review_vocab和rating_vocab

代码有详细中文注释，具体查看 **ipynb文件**


#### Re
《Natural Language Processing with Pytorch》第四章
