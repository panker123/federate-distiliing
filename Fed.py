
#服务器端将w权重进行平均计算
import numpy as np
import copy
import torch
from torch import nn

def FedAvg(w):
    #首先将第一个客户端的参数进行复制
    w_avg=copy.deepcopy(w[0])
    #遍历第一个客户端的key,并加上后面客户对应的key
    for k in w_avg.keys():
        #遍历后面的客户端
        for i in range(1,len(w)):
            w_avg[k]+=w[i][k]
        w_avg[k]=torch.div(w_avg[k],len(w))

    return w_avg


if __name__=="__main__":
    print()