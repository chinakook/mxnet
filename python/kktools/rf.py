# coding: utf-8

from __future__ import absolute_import

import json
import math
import collections

def filter_rf(k, s, p, d=1):
    k2 = (k - 1) * d + 1
    return k2, s, (k2-1)/2. - p

def deconv_rf(k, u, c):
    beta = (2 * c - k + 1) / (2. * u)
    return (k - 1)/float(u) + 1, 1 / float(u), beta

def compose_rfs(rfb, sb, ob, rft, st, ot):
    s = sb * st
    offset = sb * ot + ob
    rfsize = sb * (rft - 1) + rfb
    return rfsize, s, offset

def overlay_rfs(rf0, s0, o0, rf1, s1, o1):
    assert s0 == s1, "two strides must be the same."
    s = s0
    a = min(o0 - (rf0 - 1)/2., o1 - (rf1 - 1)/2.)
    b = max(o0 + (rf0 - 1)/2., o1 + (rf1 - 1)/2.)
    return b - a + 1, s, (a + b) / 2.

def looks_like_weight(name):
    """Internal helper to figure out if node should be hidden with `hide_weights`.
    """
    weight_like = ('_weight', '_bias', '_beta', '_gamma',
                    '_moving_var', '_moving_mean', '_running_var', '_running_mean')
    return name.endswith(weight_like)

def rf_summery(sym):
    rfs = collections.OrderedDict()

    conf = json.loads(sym.tojson())
    nodes = conf["nodes"]

    hidden_nodes = set()
    for node in nodes:
        op = node["op"]
        name = node["name"]
        if op == "null" and looks_like_weight(node["name"]):
            hidden_nodes.add(node["name"])
        elif op == "null" and name == "data":
            node["meta"] = {}
            node["meta"]["rf"] = 1
            node["meta"]["stride"] = 1
            node["meta"]["offset"] = 0
        elif op == "Convolution" or op == "Pooling":
            k = int(node["attrs"]["kernel"][1])
            if "stride" in node["attrs"]:
                s = int(node["attrs"]["stride"][1])
            else:
                s = 1
            if "pad" in node["attrs"]:
                p = int(node["attrs"]["pad"][1])
            else:
                p = 0
            if "dilate" in node["attrs"]:
                d = int(node["attrs"]["dilate"][1])
            else:
                d = 1
            rf, stride, offset = filter_rf(k,s,p,d)
            node["meta"] = {}
            node["meta"]["rf"] = rf
            node["meta"]["stride"] = stride
            node["meta"]["offset"] = offset
        elif op == "Activation" or op == "BatchNorm":
            node["meta"] = {}
            node["meta"]["rf"] = 1
            node["meta"]["stride"] = 1
            node["meta"]["offset"] = 0
        elif op == "Deconvolution":
            k = int(node["attrs"]["kernel"][1])
            u = int(node["attrs"]["stride"][1])
            c = int(node["attrs"]["pad"][1])
            rf, stride, offset = deconv_rf(k,u,c)
            node["meta"] = {}
            node["meta"]["rf"] = rf
            node["meta"]["stride"] = stride
            node["meta"]["offset"] = offset
        elif op == "UpSampling":
            scale = int(node["attrs"]["scale"])
            k = 2 * scale - scale % 2
            u = scale
            c = int(math.ceil((scale - 1) / 2.))
            rf, stride, offset = deconv_rf(k,u,c)
            node["meta"] = {}
            node["meta"]["rf"] = rf
            node["meta"]["stride"] = stride
            node["meta"]["offset"] = offset
        else:
            node["meta"] = {}
            node["meta"]["rf"] = 1
            node["meta"]["stride"] = 1
            node["meta"]["offset"] = 0

    for node in nodes:
        op = node["op"]
        name = node["name"]
        if name in hidden_nodes:
            continue
        else:
            inputs = node["inputs"]
            input_nodes = []
            for item in inputs:
                input_node = nodes[item[0]]
                if input_node["name"] in hidden_nodes:
                    continue
                input_nodes.append(input_node)
            
            if op in ("Convolution", "Pooling", "Deconvolution", "UpSampling", "Activation", "BatchNorm", "slice", "slice_axis"):
                assert len(input_nodes) == 1, "Filter layer inputs count should be 1."
                rf0 = input_nodes[0]["meta"]["rf"]
                stride0 = input_nodes[0]["meta"]["stride"]
                offset0 = input_nodes[0]["meta"]["offset"]

                rf = node["meta"]["rf"]
                stride = node["meta"]["stride"]
                offset = node["meta"]["offset"]

                rf_c, stride_c, offset_c = compose_rfs(rf0, stride0, offset0, rf, stride, offset)
                node["meta"]["rf"] = rf_c
                node["meta"]["stride"] = stride_c
                node["meta"]["offset"] = offset_c

                rfs[name] = (rf_c, stride_c, offset_c)
            elif op in ("broadcast_add", "elemwise_add", "ElementWiseSum", "add_n", "Crop", "Concat"):
                assert len(input_nodes) == 2, "ElemetWise layer inputs count should be 2."
                rf0 = input_nodes[0]["meta"]["rf"]
                stride0 = input_nodes[0]["meta"]["stride"]
                offset0 = input_nodes[0]["meta"]["offset"]
                rf1 = input_nodes[1]["meta"]["rf"]
                stride1 = input_nodes[1]["meta"]["stride"]
                offset1 = input_nodes[1]["meta"]["offset"]

                rf_over, stride_over, offset_over = overlay_rfs(rf0, stride0, offset0, rf1, stride1, offset1)
                node["meta"]["rf"] = rf_over
                node["meta"]["stride"] = stride_over
                node["meta"]["offset"] = offset_over

                rfs[name] = (rf_over, stride_over, offset_over)
    return rfs
