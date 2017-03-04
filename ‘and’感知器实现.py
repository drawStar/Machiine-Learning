#coding=utf-8
#感知器的实现
import os

class Perceptron(object):
    def __init__(self,input_num,activator):
    #初始化感知器，
    #input_num输入参数的个数
    #activator 激活函数，激活函数的类型为double->double
        self.activator=activator
        self.weights=[0.0 for _ in range(input_num)]#weights初始化为input_num个0.0
        self.bias=0.0
    def __str__(self):
    #打印学习的权重和偏置项
        return 'weights\t:%s \nbias\t:%f'%(self.weights,self.bias)

    def predict(self,input_vec):
    #输入向量，输出感知器的结果
    #zip变成[(x1,w1),(x2,w2),(x3,w3),...]
    #reduce初始值为0.0
        return self.activator(
            reduce(lambda a, b: a + b,
                   map(lambda (x, w): x * w, zip(input_vec, self.weights)) , 0.0) + self.bias)

    def train(self,input_vecs,labels,epoch,learningrate):
        for i in range(epoch):
            self.one_epoch(input_vecs,labels,learningrate)

    def one_epoch(self,input_vecs,labels,learningrate):
        # 把输入和输出打包在一起，成为样本的列表[(input_vec, label), ...]
        samples = zip(input_vecs, labels)
        # for each 样本：
        for (input_vec, label) in samples:
            output = self.predict(input_vec)
            # 更新权重
            self.update_weights(input_vec, output, label, learningrate)

    def update_weights(self,input_vec,output,label,learningrate):
    #更新weights与bias
        delta=label-output
        self.weights=map(lambda (x,w):w+learningrate*delta*x,
                         zip(input_vec,self.weights))
        self.bias+=learningrate*delta

def f(x):
#定义激活函数
    return 1 if x>0 else 0

def get_training_dataset():
#基于and真值表构建训练数据
    # 构建训练数据
    input_vecs = [[1,1], [0,0], [1,0], [0,1]]
    labels = [1, 0, 0, 0]
    return input_vecs, labels
def train_and_perceptron():
    p=Perceptron(2,f)
    #输入参数个数为2（因为and是二元函数），激活函数为f
    input_vecs,labels=get_training_dataset()
    p.train(input_vecs,labels,100,0.1)
    return p#返回训练好的感知器

if __name__ == '__main__':
    and_perception = train_and_perceptron()
    # 打印训练获得的权重
    print and_perception
    # 测试
    print '1 and 1 = %d' % and_perception.predict([1, 1])
    print '0 and 0 = %d' % and_perception.predict([0, 0])
    print '1 and 0 = %d' % and_perception.predict([1, 0])
    print '0 and 1 = %d' % and_perception.predict([0, 1])
