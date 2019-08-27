# -*- coding: utf-8 -*-
import numpy as np
import chainer
import chainer.functions as F
import chainer.links as L
import chainer.initializers as I
from chainer import training
from chainer.training import extensions

class MyChain(chainer.Chain):
    def __init__(self):
        super(MyChain, self).__init__()
        with self.initscope():
            self.l1 = L.Linear(2, 3) # 入力2, 中間層3
            self.l2 = L.Linear(3, 2) # 中間層3, 出力2
    def __call__(self, x):
        
