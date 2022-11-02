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


def one2many(x, probs, size, ispositive=True):
    dataset_elements = []
    d_input = "SOS {} EOS".format(x)
    for _ in range(size):
        sample = get_sample(probs)
        sample = [str(s) for s in sample]
        sample = " ".join(sample)
        d_output = "SOS {} EOS".format(sample)
        dataset_elements.append((d_input, d_output, str(ispositive)))

    dataset = tf.data.Dataset.from_tensor_slices(dataset_elements)
    return dataset

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

# two datapoints
# "SOS x EOS" -> ["SOS a EOS", "SOS b EOS", "SOS c EOS"]
# "SOS y EOS" -> ["SOS d EOS", "SOS e EOS", "SOS f EOS"]
def exp1(outdir):
    in1 = "SOS x EOS"
    out1 = ["SOS a EOS", "SOS b EOS", "SOS c EOS"]
    in2 = "SOS y EOS"
    out2 = ["SOS d EOS", "SOS e EOS", "SOS f EOS"]
    data = {"input":[in1, in2], "output": [out1, out2]}
    data["output"] = tf.ragged.stack(data["output"])
    dataset = tf.data.Dataset.from_tensor_slices(data)
    tf.data.experimental.save(dataset, outdir)

# exp1(outdir="synthetic/syn1/pos")

# several datapoints with the same input
def exp2(outdir, alternatives=0, constraints=10):
    data = {"input":[], "output": []}
    inp = "SOS x EOS"
    good = "SOS a EOS"
    for i in range(constraints):
        prefix = chr(ord('a')+i+1)
        out = ["SOS " + prefix + str(j) + " EOS" for j in range(alternatives)]
        for j in range(constraints):
            almost_good = "SOS a" + str(j) + " EOS"
            if j != i:
                out.append(almost_good)
        out.append(good)
        print("out: ", out)
        data["input"].append(inp)
        data["output"].append(out)
        
    data["output"] = tf.ragged.stack(data["output"])
    dataset = tf.data.Dataset.from_tensor_slices(data)
    tf.data.experimental.save(dataset, outdir)

# exp2(outdir="synthetic/syn2_10/pos", alternatives=10, constraints=10)
# exp2(outdir="synthetic/syn2_50/pos", alternatives=10, constraints=50)
# exp2(outdir="synthetic/syn2_100/pos", alternatives=10, constraints=100)

# several datapoints with the same input and conflicting outputs
def exp3(outdir):
    inp = "SOS x EOS"
    out = [
        ["a", "b", "c"],
        ["a", "b", "d"],
        ["a", "b", "e"],
        ["a", "c", "d"],
        ["c", "e", "f"]
    ]
    out = [["SOS " + x + " EOS" for x in o] for o in out]

    print("out: ", out)
    
    data = {"input":[inp] * len(out), "output": out}        
    data["output"] = tf.ragged.stack(data["output"])
    dataset = tf.data.Dataset.from_tensor_slices(data)
    tf.data.experimental.save(dataset, outdir)

# two datapoints
# "SOS x EOS" -> ["SOS a EOS", "SOS b EOS", "SOS c EOS"]
# "SOS y EOS" -> ["SOS d EOS", "SOS e EOS", "SOS f EOS"]
def exp4(outdir):
    in1 = "SOS EOS"
    out1 = ["SOS 1 1 EOS EOS", "SOS 1 2 1 EOS EOS", "SOS 1 2 2 EOS EOS", "SOS 2 EOS EOS"]
    in2 = "SOS EOS"
    out2 = ["SOS 1 1 EOS EOS", "SOS 1 2 1 EOS EOS", "SOS 1 2 2 EOS EOS", "SOS 2 EOS EOS"]
    data = {"input":[in1, in2], "output": [out1, out2]}
    data["output"] = tf.ragged.stack(data["output"])
    dataset = tf.data.Dataset.from_tensor_slices(data)
    tf.data.experimental.save(dataset, outdir)


exp4(outdir="synthetic/syn4/pos")

# synthetic data for sequences
# slen: sequence length
# ntoken: number of tokens
# ncand: number of candidates per constraint
# nconst: number of constraints per input
# nsample: number of positive/negative samples
# outdir: directory to save data
# def seq_generator(slen, ntoken, ncand, nconst, nsample, outdir):
#     tokens = [str(i) for i in range(ntoken)]

#     def get_random_output(tokens, slen):
#         curr_tokens = random.choices(tokens, k=slen)
#         o = "SOS {} EOS".format(" ".join(curr_tokens))
#         return o


#     # get one output that satisfies all constraints
#     good_outputs = [get_random_output(tokens, slen) for _ in range(nsample)]
        
#     for ispositive, pname in zip((True, False), ("pos", "neg")):
#         inputs = []
#         outputs = []
#         for i in range(nsample):
#             inp = "SOS {} EOS".format(i)
#             for _ in range(nconst):
#                 curr_out = []
#                 for c in range(ncand):
#                     if ispositive:
#                         if c == 0: # the first candidate is going to be a good one
#                             o = good_outputs[i]
#                         else:
#                             o = get_random_output(tokens, slen)
#                     else: # negatives should not have the good one as candidate
#                         while True:
#                             o = get_random_output(tokens, slen)
#                             if o != good_outputs[i]:
#                                 break                            
#                     curr_out.append(o)
                    
#                 inputs.append(inp)
#                 outputs.append(curr_out)
            
#         path = "{}/{}".format(outdir, pname)
#         outputs = tf.ragged.stack(outputs)
#         dataset = tf.data.Dataset.from_tensor_slices({"input":inputs, "output":outputs})
#         print("Element spec for {}:{}".format(path, dataset.element_spec))
#         tf.data.experimental.save(dataset, path)
#         print("   SAVED TO: ", path)

# def generate_sequences():
#     ntoken=10
#     nconst = 5
#     nsample = 100
#     for slen in (1,2,3,4,5,6,7,8,9,10):
#         ncand = ((slen // 3) + 1) * 5
#         outdir = "synthetic/sec_100/seclen{}".format(slen)
#         seq_generator(slen=slen,
#                       ntoken=ntoken,
#                       ncand=ncand,
#                       nconst=nconst,
#                       nsample=nsample,
#                       outdir=outdir
#         )
                      
# generate_sequences()


# seq_generator(slen=3,
#               ntoken=10,
#               ncand=4,
#               nconst=4,
#               nsample=5,
#               outdir="synthetic/sec/sec1"
# )


    
# probs1 = [0.6, 0.3, 0.09, 0.01]
# path1 = "synthetic/syn1"
# size1=100000
# one2one(probs1, size1, path1)


# leaf1 = [(0.6,[]), (0.3,[]), (0.09,[]), (0.01,[])]
# leaf2 = [(0.5,[]), (0.3,[]), (0.1,[]), (0.1,[])]
# leaf3 = [(0.4,[]), (0.2,[]), (0.2,[]), (0.2,[])]
# leaf4 = [(0.8,[]), (0.1,[]), (0.05,[]), (0.05,[])]
# probs2 = [(0.6,leaf1), (0.3,leaf2), (0.09,leaf3), (0.01,leaf4)]
# size2=100000
# path2 = "synthetic/syn2"
# one2many(probs2, size2, path2)



# SIZE = 100000


# p1 = [0.6, 0.3, 0.09, 0.01]
# p2 = [0.5, 0.3, 0.1, 0.1]
# p3 = [0.4, 0.2, 0.2, 0.2]
# p4 = [0.8, 0.1, 0.05, 0.05]

# deg11 = create_leaf(p1)
# deg12 = create_leaf(p2)
# deg13 = create_leaf(p3)
# deg14 = create_leaf(p4)

# deg21 = create_node(p1, [deg11, deg12, deg13, deg14])
# deg22 = create_node(p2, [deg12, deg13, deg14, deg11])
# deg23 = create_node(p3, [deg13, deg14, deg11, deg12])
# deg24 = create_node(p4, [deg14, deg11, deg12, deg13])

# deg31 = create_node(p1, [deg21, deg22, deg23, deg24])

# d1 = one2many("deg11", deg11, SIZE)
# d2 = one2many("deg12", deg12, SIZE)
# d3 = one2many("deg21", deg21, SIZE)
# d4 = one2many("deg22", deg22, SIZE)
# d5 = one2many("deg31", deg31, SIZE)

# dataset = d1
# for d in (d2, d3, d4, d4):
#     dataset = dataset.concatenate(d)

# tf.data.experimental.save(dataset, "synthetic/combined")

# if False:
#     d1 = one2many("deg11", deg11, SIZE//1000)
#     d2 = one2many("deg12", deg12, SIZE//100)
#     d3 = one2many("deg21", deg21, SIZE//10)
#     d4 = one2many("deg22", deg22, SIZE//10)
#     d5 = one2many("deg31", deg31, SIZE)
#     dataset = d1
#     for d in (d2, d3, d4, d4):
#         dataset = dataset.concatenate(d)
#     tf.data.experimental.save(dataset, "synthetic/combined_unbalanced")

# if False:
#     SIZE=10000
    
#     p1 = [0.4, 0.3, 0.2, 0.1]
#     p2 = [0.2, 0.0, 0.5, 0.3]

#     deg11 = create_leaf(p1)
#     deg12 = create_leaf(p2)
#     d1 = one2many("input", deg11, SIZE, ispositive=True)
#     d2 = one2many("input", deg12, SIZE, ispositive=False)
#     dataset=d1
#     datasejt=dataset.concatenate(d2)
#     print("Dataset size:", tf.data.experimental.cardinality(dataset).numpy())
#     tf.data.experimental.save(dataset, "synthetic/len1_neg")

# if False: # four sequences: 0, 1, 23, 24
#     SIZE=10000
#     deg11 = create_leaf([0.6, 0.4, 0, 0, 0])
#     d1 = one2many("input", deg11, SIZE, ispositive=True)

#     deg12 = create_leaf([0.0, 0.0, 0, 0.6, 0.4])
#     deg21 = create_node([0.0, 0.0, 1.0, 0.0, 0.0], [deg12, deg12, deg12, deg12, deg12])    
#     d2 = one2many("input", deg21, SIZE, ispositive=True)
    
#     dataset=d1
#     dataset=dataset.concatenate(d2)
#     print("Dataset size:", tf.data.experimental.cardinality(dataset).numpy())
#     tf.data.experimental.save(dataset, "synthetic/len_effect1")

# if False: # four sequences: 0, 1, 20, 21
#     SIZE=10000
#     deg11 = create_leaf([0.6, 0.4, 0, 0, 0])
#     d1 = one2many("input", deg11, SIZE, ispositive=True)

#     deg21 = create_node([0.0, 0.0, 1.0, 0.0, 0.0], [deg11, deg11, deg11, deg11, deg11])
#     d2 = one2many("input", deg21, SIZE, ispositive=True)
    
#     dataset=d1
#     dataset=dataset.concatenate(d2)
#     print("Dataset size:", tf.data.experimental.cardinality(dataset).numpy())
#     tf.data.experimental.save(dataset, "synthetic/len_effect2")


# if False: # four sequences: 0, 1, 2, 3
#     SIZE=20000 
#     p1 = [0.3, 0.2, 0.3, 0.2]
#     deg11 = create_leaf(p1)
#     d1 = one2many("input", deg11, SIZE, ispositive=True)
#     dataset=d1
#     print("Dataset size:", tf.data.experimental.cardinality(dataset).numpy())
#     tf.data.experimental.save(dataset, "synthetic/len_effect3")

# if False: # 1 sequence: 0
#     SIZE=20000 
#     p1 = [0.0, 0.0, 0.0, 0.0]
#     deg11 = create_leaf(p1)
#     d1 = one2many("input", deg11, SIZE, ispositive=True)
#     dataset=d1
#     print("Dataset size:", tf.data.experimental.cardinality(dataset).numpy())
#     tf.data.experimental.save(dataset, "synthetic/len_effect4")

# if True:
#     # positives: 100 * [0, 1]
#     # negatives: 10 * [1, 2], 90 * [2]

#     d_input = "SOS input EOS"
#     p0 = "SOS 0 EOS"
#     p1 = "SOS 1 EOS"
#     p2 = "SOS 2 EOS"
#     d_output0 = [p0, p1]
#     d_output1 = [p1, p2]
#     d_output2 = [p2]
    
#     elements = {
#         True: {"input": [], "output": []},
#         False: {"input": [], "output": []},
#     }

#     for _ in range(100):
#         elements[True]["input"].append(d_input)
#         elements[True]["output"].append(d_output0)
#     for _ in range(10):
#         elements[False]["input"].append(d_input)
#         elements[False]["output"].append(d_output1)
#     for _ in range(90):
#         elements[False]["input"].append(d_input)
#         elements[False]["output"].append(d_output2)

#     path_pos = "synthetic/select/pos"
#     path_neg = "synthetic/select/neg"
#     for ispositive, p in zip((True, False), (path_pos, path_neg)):
#         elements[ispositive]["output"] = tf.ragged.stack(elements[ispositive]["output"])
#         dataset = tf.data.Dataset.from_tensor_slices(elements[ispositive])
#         print("Element spec for {} {}:{}".format(p, ispositive, dataset.element_spec))
#         tf.data.experimental.save(dataset, p)
