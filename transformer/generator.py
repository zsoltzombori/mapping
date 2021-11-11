import tensorflow as tf
import random
import numpy as np


def one2one(probs, size, path, ispositive=True):
    dataset_elements = []
    d_input = "SOS X EOS"
    outputs = random.choices(range(len(probs)), weights=probs, k=size)
    for o in outputs:
        d_output = "SOS {} EOS".format(o)
        dataset_elements.append((d_input, d_output, str(ispositive)))

    dataset = tf.data.Dataset.from_tensor_slices(dataset_elements)
    tf.data.experimental.save(dataset, path)


def one2many(probs, size, path, ispositive=True):
    dataset_elements = []
    d_input = "SOS X EOS"
    for _ in range(size):
        sample = get_sample(probs)
        sample = [str(s) for s in sample]
        sample = " ".join(sample)
        d_output = "SOS {} EOS".format(sample)
        dataset_elements.append((d_input, d_output, str(ispositive)))

    dataset = tf.data.Dataset.from_tensor_slices(dataset_elements)
    tf.data.experimental.save(dataset, path)

def get_sample(probs):
    if probs == []:
        return []

    curr_probs = [p[0] for p in probs]
    result = random.choices(range(len(probs)), weights=curr_probs, k=1)[0]
    next_probs = probs[result][1]
    result = [result] + get_sample(next_probs)
    return result

def create_leaf(probs):
    result = [(p, []) for p in probs]
    return result

def create_node(probs, nodes):
    assert len(probs) == len(nodes)
    result = [(p,n) for (p,n) in zip(probs, nodes)]
    return result


    
probs1 = [0.6, 0.3, 0.09, 0.01]
path1 = "synthetic/syn1"
size1=100000
one2one(probs1, size1, path1)


leaf1 = [(0.6,[]), (0.3,[]), (0.09,[]), (0.01,[])]
leaf2 = [(0.5,[]), (0.3,[]), (0.1,[]), (0.1,[])]
leaf3 = [(0.4,[]), (0.2,[]), (0.2,[]), (0.2,[])]
leaf4 = [(0.8,[]), (0.1,[]), (0.05,[]), (0.05,[])]
probs2 = [(0.6,leaf1), (0.3,leaf2), (0.09,leaf3), (0.01,leaf4)]
size2=100000
path2 = "synthetic/syn2"
one2many(probs2, size2, path2)






p1 = [0.6, 0.3, 0.09, 0.01]
p2 = [0.5, 0.3, 0.1, 0.1]
p3 = [0.4, 0.2, 0.2, 0.2]
p4 = [0.8, 0.1, 0.05, 0.05]

deg11 = create_leaf(p1)
deg12 = create_leaf(p2)
deg13 = create_leaf(p3)
deg14 = create_leaf(p4)

deg21 = create_node(p1, [deg11, deg12, deg13, deg14])
deg22 = create_node(p2, [deg12, deg13, deg14, deg11])
deg23 = create_node(p3, [deg13, deg14, deg11, deg12])
deg24 = create_node(p4, [deg14, deg11, deg12, deg13])

probs3 = create_node(p1, [deg21, deg22, deg23, deg24])
size3=100000
path3 = "synthetic/syn3"
one2many(probs3, size3, path3)
