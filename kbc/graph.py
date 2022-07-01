import os
import sys
import numpy as np
import itertools
import random
# import tensorflow as tf
import pickle


class Graph():

    def __init__(self, filename):
        file1 = open(filename, 'r')
        Lines = file1.readlines()

        self.edgedict = {}
        self.nodes = []
        self.preds = []
        
        for line in Lines:
            triple = line.strip().split()
            head, pred, tail = triple

            inv_pred = self.get_inv(pred)

            self.add_edge(head, pred, tail)
            self.add_edge(tail, inv_pred, head)

            if head not in self.nodes:
                self.nodes.append(head)
            if tail not in self.nodes:
                self.nodes.append(tail)
            if pred not in self.preds:
                self.preds.append(pred)
            if inv_pred not in self.preds:
                self.preds.append(inv_pred)

    def get_inv(self, pred):
        if pred.startswith("INV_"):
            return pred[:4]
        else:
            return "INV_" + pred

    def triple2key(self, triple):
        return tuple(triple)


    def add_edge(self, head, pred, tail):
        if head not in self.edgedict:
            self.edgedict[head] = {}
        if pred not in self.edgedict[head]:
            self.edgedict[head][pred] = []
        if tail not in self.edgedict[head][pred]:
            self.edgedict[head][pred].append(tail)

    def find_path(self, head, tail, k, node_history, exclude=[]):
        if head == tail:
            return [[]]
        if k == 0:
            return []
        if head in node_history:
            return []
        if head not in self.edgedict:
            return []
            
        result = []
        node_history2 = node_history + [head]
        for pred in self.edgedict[head]:
            for next_head in self.edgedict[head][pred]:
                if (head, pred, next_head) in exclude:
                    continue
                else:
                    curr_paths = self.find_path(next_head, tail, k-1, node_history2, exclude)
                    curr_paths = [p + [pred] for p in curr_paths]
                    curr_paths.sort()
                    curr_paths = list(curr_paths for curr_paths,_ in itertools.groupby(curr_paths))
                    result += curr_paths                
        return result

    def find_path_excluding(self, head, pred, tail, k):
        inv_pred = self.get_inv(pred)
        exclude = [(head, pred, tail), (tail, inv_pred, head)]
        
        paths = self.find_path(head, tail, k, [], exclude)
        return paths

    # corruption in ["head", "tail", "both"]
    def generate_negatives(self, positives, neg_cnt, corruption="both"):
        negative_dict = {}
        negatives = []
        for head, pred, tail in positives:
            negatives_curr = []
            while len(negatives_curr) < neg_cnt:
                
                if corruption == "both":
                    target = random.choice(["head", "tail"])
                else:
                    target = corruption
                    
                if target == "head":
                    head2 = random.choice(self.nodes)
                    negative = [head2, pred, tail]
                elif target == "tail":
                    tail2 = random.choice(self.nodes)
                    negative = [head, pred, tail2]

                if (negative not in positives) and (negative not in negatives_curr):
                    negatives_curr.append(negative)

            negatives += negatives_curr
            key = self.triple2key([head, pred, tail])
            negative_dict[key] = negatives_curr
        return negatives, negative_dict
                    



class EdgeList():
    def __init__(self, filename):
        file1 = open(filename, 'r')
        Lines = file1.readlines()

        self.triples = []
        for line in Lines:
            triple = line.strip().split()
            self.triples.append(triple)

def elements2file(elements, path):        
    path_pos = "{}/pos".format(path)
    path_neg = "{}/neg".format(path)

    for ispositive, p in zip((True, False), (path_pos, path_neg)):
        if len(elements[ispositive]["output"]) > 0:
            elements[ispositive]["output"] = tf.ragged.stack(elements[ispositive]["output"])
            dataset = tf.data.Dataset.from_tensor_slices(elements[ispositive])
            # print("Element spec for {} {}:{}".format(p, ispositive, dataset.element_spec))
            tf.data.experimental.save(dataset, p)
            print("   SAVED TO ", p)
        else:
            print("Empty elements for {} {}".format(p, ispositive))

def edges2file(graph, edgefile, neg_cnt, corruption, pathlen, outdir):
    positives = EdgeList(edgefile).triples
    negatives, negative_dict = graph.generate_negatives(positives, neg_cnt, corruption)

    elements = {
        True: {"input": [], "output": []},
        False: {"input": [], "output": []},
    }

    for triples, ispositive in zip((positives, negatives), (True, False)):
        for head, pred, tail in triples:
            d_input = " ".join(["SOS", pred, "PREDEND", head, tail, "EOP", "EOS"])
            paths = graph.find_path_excluding(head, pred, tail, pathlen)
            if len(paths) == 0:
                continue
            d_output = ["SOS " + " ".join(p) + " EOS" for p in paths]
            elements[ispositive]["input"].append(d_input)
            elements[ispositive]["output"].append(d_output)

    elements2file(elements, outdir)
    

        
def create_dataset(graphfile, trainfile, devfile, testfile, neg_cnt, corruption, pathlen, outdir):
    graph = Graph(graphfile)
    edges2file(graph, trainfile, neg_cnt, corruption, pathlen, outdir+"/train")
    edges2file(graph, devfile, neg_cnt, corruption, pathlen, outdir+"/dev")
    edges2file(graph, testfile, neg_cnt, corruption, pathlen, outdir+"/test")

def create_data_object(graphfile, trainfile, devfile, testfile, neg_cnt, corruption, pathlen, outdir):
    os.makedirs(outdir, exist_ok=True)
    graph = Graph(graphfile)
    for edgefile, name in zip((trainfile, devfile, testfile), ("train", "dev", "test")):
        positives = EdgeList(edgefile).triples
        # positives = positives[:10] # todo REMOVE
        negatives, posneg_dict = graph.generate_negatives(positives, neg_cnt, corruption)

        pos_dict = {}
        for head, pred, tail in positives:
            pos_dict[(head, pred, tail)] = graph.find_path_excluding(head, pred, tail, pathlen)
        neg_dict = {}
        for head, pred, tail in negatives:
            neg_dict[(head, pred, tail)] = graph.find_path_excluding(head, pred, tail, pathlen)
        
        result = {"pos_dict":pos_dict, "neg_dict":neg_dict, "posneg_dict":posneg_dict}
        outfile = "{}/{}".format(outdir, name)
        with open(outfile, 'wb') as f:
            pickle.dump(result, f)

experiment = "WN18RR"
# experiment = "FB15K-237"

graphfile = "datasets/{}/graph.txt".format(experiment)
trainfile = "datasets/{}/train.txt".format(experiment)
devfile = "datasets/{}/dev.txt".format(experiment)
testfile = "datasets/{}/test.txt".format(experiment)
outdir = "out/{}".format(experiment)

create_data_object(graphfile, trainfile, devfile, testfile, neg_cnt=50, corruption="both", pathlen=4, outdir=outdir)

