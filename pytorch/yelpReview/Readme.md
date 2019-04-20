pytorch实现情感分类
根据餐厅评论推测积极情绪还是消极情绪

#### Dataset

数据集为yelp数据集，下载后解压到data文件夹
百度网盘链接：
链接：https://pan.baidu.com/s/1bpuiqe1Sram0sqGjPNlprA 
提取码：sm9f 

#### Abstract

embedding使用的collapsed one-hot表示
Model为一层nn.Linear()线性模型
BCEWithLogitsLoss 实现 二分类


主要涉及三个辅助类：
Vocabulary ： mapping text tokens to intergers and mapping the class labels to integers   .
ReviewDataset : 包含DataFrame格式的数据集，和ReviewVectorizer类的实例化vectorizer
ReviewVectorizer： 将word转为向量，包括review_vocab和rating_vocab

代码有详细中文注释，具体查看 **yelpReview.ipynb**


#### Re
《Natural Language Processing with Pytorch》第三章
