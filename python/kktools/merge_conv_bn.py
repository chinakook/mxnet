import numpy as np
import mxnet as mx
import json


def merge(sym, args, auxs):
        
    graph=json.loads(sym.tojson())
    nodes = graph['nodes']
    args_deploy = {}
    #auxs_deploy = {}

    ignore_bn = {}
    ignore_bn_args = set()

    for i, n in enumerate(nodes):
        if n['op'] == 'BatchNorm':
            pre_layer =  nodes[n['inputs'][0][0]]
            if pre_layer['op'] == 'Convolution':
                ignore_bn[pre_layer['name']] = i
                ignore_bn_args.add(n['inputs'][1][0])
                ignore_bn_args.add(n['inputs'][2][0])
                ignore_bn_args.add(n['inputs'][3][0])
                ignore_bn_args.add(n['inputs'][4][0])
        if len(n['inputs']) > 0:
            print(i + len(n['inputs']))


    for i, n in enumerate(nodes):
        if n['op'] == 'Convolution' and n['name'] in ignore_bn:
            bn = nodes[ignore_bn[n['name']]]
            
            gamma = args[nodes[bn['inputs'][1][0]]['name']]
            beta = args[nodes[bn['inputs'][2][0]]['name']]
            moving_mean = auxs[nodes[bn['inputs'][3][0]]['name']]
            moving_var = auxs[nodes[bn['inputs'][4][0]]['name']]
            eps = float(bn['attrs']['eps'])

            weight = args[nodes[n['inputs'][1][0]]['name']]
            if 'no_bias' not in n['attrs'] or n['attrs']['no_bias'] == 'False':
                bias = args[nodes[n['inputs'][2][0]]['name']]
            else:
                bias = mx.nd.zeros((weight.shape[0],))              
            a = gamma / mx.nd.sqrt(moving_var + eps)
            b = beta - a * moving_mean
            a = mx.nd.reshape(a,(-1,1,1,1))
            weight = weight * a
            bias = bias + b
            
            args_deploy[nodes[n['inputs'][1][0]]['name']] = weight
            args_deploy[n['name']+'_bias'] = bias
        else:
            for widx in n['inputs']:
                if widx[0] not in ignore_bn_args \
                    and nodes[widx[0]]['name'].endswith(('_weight', '_bias', '_beta', '_gamma')):
                    args_deploy[nodes[widx[0]]['name']] = args[nodes[widx[0]]['name']]

    return args_deploy