# coding: utf-8

from __future__ import absolute_import

import sys
import numpy as np
sys.path.insert(0, '../mxnet')

import mxnet as mx

def statparams(sym, **kwargs):
    arg_shapes, _, aux_shapes = sym.infer_shape(**kwargs)
    arg_names = sym.list_arguments()
    aux_names = sym.list_auxiliary_states()
    arg_shape_dic = dict(zip(arg_names, arg_shapes))
    #aux_shape_dic = dict(zip(aux_names, aux_shapes))

    total = 0
    for k, v in arg_shape_dic.items():
        if k == 'data':
            continue
        total += np.prod(v)

    return total

@mx.init.register
class MyConstant(mx.init.Initializer):
    def __init__(self, value):
        super(MyConstant, self).__init__(value=value)
        self.value = value

    def _init_weight(self, _, arr):
        arr[:] = mx.nd.array(self.value)

def get_shape(x):
    if isinstance(x, mx.nd.NDArray):
        return x.shape
    elif isinstance(x, mx.symbol.Symbol):
        _,x_shape,_=x.infer_shape_partial()
        return x_shape[0] 