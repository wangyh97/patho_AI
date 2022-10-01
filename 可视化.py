from this import d
import numpy as np
import matplotlib.pyplot as plt

def draw(training,validation,test,fold):
    plt.title('dataset split')
    N = training.size
    ind = np.arange(N)
    plt.xticks(N,np.arange(N))
    plt.yticks(np.arange(0,81,20))
    plt.ylabel('percentage')

    sum = training+validation+test
    bottom = training/sum
    center = validation/sum
    top = test/sum

    width = 0.35  # 设置条形图一个长条的宽度

    p1color = ['blue' if x else 'royalblue' for x in fold]
    p2color = ['green' if x else 'limegreen'for x in fold]
    p3color = ['red' if x else 'orangered' for x in fold]

    p1 = plt.bar(ind, bottom, width, color=p1color)  
    p2 = plt.bar(ind, center, width, bottom=bottom,color=p2color)  #在p1的基础上绘制，底部数据就是p1的数据
    p3 = plt.bar(ind, top, width, bottom=d,color=p3color)    #在p1和p2的基础上绘制，底部数据就是p1和p2

    plt.legend((p1[0], p2[0], p3[0]), ('training', 'validation', 'test'),loc = 3)

    plt.show()

training = np.array([70])
validation = np.array([20])
test = np.array([10])
fold = ['True']

draw(training,validation,test,fold)


