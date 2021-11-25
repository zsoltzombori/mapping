import numpy as np
import transformer
import tensorflow as tf
import copy
import argparse
parser = argparse.ArgumentParser()
parser.add_argument('--datadir', type=str, required=True)
args = parser.parse_args()

DATADIR = args.datadir
BUFFER_SIZE = 200000


class Node:
    def __init__(self, token):
        self.token = token
        self.visit = 0
        self.children = {}

    def add_sequence(self, sequence):
        self.visit += 1
        if len(sequence) == 0:
            return
        else:            
            token = sequence[0]
            if not token in self.children:
                self.children[token] = Node(token)
            tail = sequence[1:]
            self.children[token].add_sequence(tail)

    def prob(self, token):
        if token in self.children:
            return self.children[token].visit / self.visit
        else:
            return 0.0

    def child_probs(self):
        result = []
        for token in self.children:
            prob = self.children[token].visit / self.visit
            result.append((prob, token))
        return result

    def best_sequence(self):
        probs = self.child_probs()
        if len(probs) == 0:
            return 1.0, []
        
        probs.sort(reverse=True)
        best_prob, best_token = probs[0]
        print([p[0] for p in probs])
        prob, sequence = self.children[best_token].best_sequence()
        prob = prob * best_prob
        sequence.insert(0, best_token)
        return prob, sequence

def dataset2tree(dataset):
    pos_root = Node("")
    neg_root = Node("")
    for d in dataset:
        output = d[1].numpy().decode('UTF-8').split()
        ispositive = ("True" == d[2].numpy().decode('UTF-8'))

        input = d[0].numpy().decode('UTF-8')
        output.insert(0, input)

        if ispositive:
            pos_root.add_sequence(output)
        else:
            neg_root.add_sequence(output)
    return pos_root, neg_root

def beam_search(root, critique_root, beamsize, maxlen):

    t = {
        "prob":1.0,
        "node": root,
        "sequence": [],
        "ended":False,
        "prob_c": 1.0,
        "node_c": critique_root,
    }
    top = [t]
    
    while len(top) > 0:

        # expand most probable sequence that hasn't ended yet
        index = 0
        found = False
        while index < len(top):
            curr = top[index]
            if not curr["ended"]:
                del top[index]
                found = True
                break
            index += 1
        if not found:
            break


        # expand the selected item
        node = curr["node"]
        critique = curr["node_c"]
        probs = node.child_probs()
        probs.sort(reverse=True)
        probs = probs[:beamsize]

        # merge the top k expansions into top
        insert_index = 0
        for prob, token in probs:
            new_prob = curr["prob"] * prob
            new_node = node.children[token]
            new_sequence = copy.copy(curr["sequence"])
            new_sequence.append(token)
            new_ended = (len(new_node.children) == 0) or (len(new_sequence) >= maxlen)

            if critique is None:
                new_prob_c = 0.0
                new_node_c = None
            else:
                new_prob_c = critique.prob(token) * curr["prob_c"]
                if token in critique.children:
                    new_node_c = critique.children[token]
                else:
                    new_node_c = None
            

            child = {
                "prob": new_prob,
                "node": new_node,
                "sequence": new_sequence,
                "ended": new_ended,
                "prob_c": new_prob_c,
                "node_c": new_node_c,
            }

            # find the insert location
            while(True):
                if insert_index >= len(top):
                    break
                else:
                    t = top[insert_index]
                    if child["prob"] > t["prob"]:
                        break
                insert_index += 1

            top.insert(insert_index, child)
        top = top[:beamsize]
    return top
        


        
# load data
(train_examples, val_examples, test_examples) = transformer.load_data(DATADIR, BUFFER_SIZE)
pos_tree, neg_tree = dataset2tree(train_examples)

for input_token in pos_tree.children:
    pos_root = pos_tree.children[input_token]
    if input_token in neg_tree.children:
        neg_root = neg_tree.children[input_token]
    else:
        neg_root = None

    tops = beam_search(pos_root, neg_root, beamsize=30, maxlen=20)
    remaining = 5
    print("\n\n----------------", input_token, "\n")
    for t in tops:
        if remaining <= 0:
            break
        rule, isvalid = transformer.parse_rule(t["sequence"])
        if isvalid and t["prob_c"] == 0:
            print(t["prob"], ": ", rule)
            remaining -= 1

#best_prob_pos, best_seq_pos = pos_tree.best_sequence()
#print(best_prob_pos, best_seq_pos)
