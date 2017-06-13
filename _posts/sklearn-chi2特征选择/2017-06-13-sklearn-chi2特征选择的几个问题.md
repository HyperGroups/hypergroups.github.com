---
layout: post
category:Computer
title:sklearn-chi2特征选择
---

sklearn-chi2特征选择的一个问题是数据量大,特征空间太大时,有些稀疏的特征出现次数为1次时会有问题,除数为0
一开始的一个想法是,找个库函数,输入数据,然后得到一个结果,
因此head一点数据来做
0 1471:1 2427:1 2570:1 3239:1 4066:1 5906:1 8280:1 8330:1 10548:1 10744:1 11627:1 12189:1 12711:1 14110:1 14913:1 16412:1 16505:1 17044:1 18465:1 20442:1 20877:1 21430:1 21566:1 21608:1 23106:1 23744:1 23821:1 23822:1 23974:1 27641:1 29063:1 29249:1 29252:1 29955:1 30804:1 30868:1 31373:1 31940:1 32498:1 32544:1 34076:1 35804:1 39147:1 39727:1 41918:1 43416:1 44357:1 44950:1 45385:1 46117:1 46438:1 46973:1 49770:1 50517:1 52079:1 53773:1 54543:1 56119:1 57528:1 57929:1 59258:1 59933:1
跑得通了,再使用完整的数据集跑一个结果
from sklearn.externals.joblib import Memory
from sklearn.datasets import load_svmlight_file
import sys
mem = Memory("./mycache")
file_name=sys.argv[1]
@mem.cache


def get_data():
    data = load_svmlight_file(file_name)
    print data
    return data[0], data[1]

X, y = get_data()

from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2

data  = SelectKBest(chi2, k=3).fit_transform(X, y)

from sklearn.datasets import dump_svmlight_file
dump_svmlight_file(data, y, file_name + '.select' ,False)

-------------
/data/machaoqun/bin/anaconda2/lib/python2.7/site-packages/sklearn/feature_selection/univariate_selection.py:165: RuntimeWarning: invalid value encountered in divide

这里可能有会遇到两个问题,一个是内存问题,一个是这个除数为0的问题,内存问题之前测试过load_svmlight_file的性能,加载及做normalize是able的,除非chi2时会占用更多的内存,但是因为这个chi2计算逻辑比较简单,我感觉不会出问题.

上面的报错一开始没整明白,反正一开始的想法不能马上实现了,就自己实现了一下chi2,
效果还行,30个G的训练数据百万级别的特征,筛选后20w的特征空间,20个G的数据量,减少了三分之一的文件大小,auc从0.86到0.85.

回过头来看,如果要更好地自己实现,可以参考这个sklearn的实现,一个是特征重编码,一个是样本剔除[筛选出最好的一个特征,如果某一行里没有这个特征,此行就会被删除].
在实际过程中,如果要使用样本权重,其实是比较蛋疼的,要考虑权重也要删除对应行,权重文件可能在某个时间点被分离出来了. 而如此,这种现成的方法就不太适用了. 而且可能还有别的问题,比如重编码的一个问题是预测的时候原始特征训练文件也得重编码,那么得有一个编码映射文件,是否能输出?又比如测试集和训练集分别进行chi2选择的话, 重编码可能是不一样的,因此chi2要在整个数据集上进行,最后再切分训练集和测试集.

1 1:1 2:1 4:1
1 1:1 2:1 4:1
0 3:1 4:1
0 5:1
1 2:1
1 10:1
1 10:1
1 10:1







