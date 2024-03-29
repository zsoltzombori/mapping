import os
import sys
import copy
# import jax
import numpy as np

EPS = 1e-8
MIN_PROB = 1e-3

def sequence_to_key(sequence, padding=False, empty=0, length=10):
    if padding:
        return tuple(sequence + (length - len(sequence))*[empty])
    else:
        return tuple(sequence)

def key_to_sequence(key):
    return list(key)


def get_seq_probs(sequences, tree):
    seq_prob_list = []
    for s in sequences:
        sequence_key = sequence_to_key(s)
        info = tree[sequence_key]
        seq_prob_list.append(info['descendant'])
    return seq_prob_list


def build_tree(sequences, prob_dict, empty=0):
    tree = {}
    seq_prob_list = []
    for s in sequences:
        # if sequence is empty -> ignore!
        if s[0] == empty:
            pass
        else:
            seq_prob = 1.0
            for i in range(len(s)+1):
                if i < len(s):
                    next_token = s[i]
                    prefix = list(s)[:i]
                else:
                    next_token = empty
                    prefix = list(s)
                inp = sequence_to_key(prefix, padding=True, empty=empty, length=len(s))
                if inp not in tree:
                    tree[inp] = {"descendant":0.0, "depths":[]}
                prob = prob_dict[tuple(inp)][next_token]
                tree[inp][next_token] = prob
                seq_prob *= prob
            for i in range(len(s)+1):
                if i < len(s):
                    next_token = s[i]
                    prefix = list(s)[:i]
                else:
                    next_token = empty
                    prefix = list(s)
                inp = sequence_to_key(prefix, padding=True, empty=empty, length=len(s))
                dkey = (next_token,"descendant")
                if dkey not in tree[inp]:
                    tree[inp][dkey] = 0.0
                tree[inp][dkey]+=seq_prob
                tree[inp]["descendant"] += seq_prob
                tree[inp]["depths"].append(len(s) - i) # TODO VERIFY DEPTH
            # print("Seq: {}, prob: {}".format(s, seq_prob))
            seq_prob_list.append(seq_prob)

    ratios = []
    for i in range(len(seq_prob_list)):
        for j in range(i+1, len(seq_prob_list)):
            ratios.append(seq_prob_list[i] / (seq_prob_list[j] + EPS))
    # print("Ratios: ", np.around(np.asarray(ratios),2))
    # print("Sum prob: ", jnp.sum(jnp.array(seq_prob_list)))
    return tree



def get_target_probs(minimums, leaves, depths, multiplier, probs):    
    minimums = np.array(minimums)
    # print("Minimums", minimums)

    leaves = np.array(leaves)
    depths = np.array(depths)

    slacks = np.equal(minimums, 0)
    nonleaves = np.equal(leaves, 0)

    # leaves should get their minimum
    leaf_targets = leaves * minimums

    # nonleaf targets start from the minimum and then get promoted
    nonleaf_targets = nonleaves * minimums

    capacity = 1 - np.sum(minimums)

    curr_multiplier = multiplier ** (1/np.maximum(1,depths))
    ideal_nonleaf_targets = (slacks * probs + (1-slacks) * curr_multiplier * probs) * nonleaves

    # print("Ideal nonleaf targets", ideal_nonleaf_targets)

    direction = np.maximum(0.0, ideal_nonleaf_targets - nonleaf_targets)
    if np.sum(direction) > 0:
        nonleaf_targets += direction / np.sum(direction) * capacity

    result = leaf_targets + nonleaf_targets
    return result


def build_update_dict(anc, anc_prob, multiplier, tree, prob_dict, token_num, empty, seq_len):
    # inp = pad_sequence(anc)
    inp = sequence_to_key(anc, padding=True, empty=empty, length=seq_len)
    # print("Key",inp)
    if inp not in tree:
        return {}
    
    curr_dict = tree[inp]
    minimums = []
    leaves = []
    depths = []
    
    # calculate target probs for current node (inp)
    for next_token in range(token_num):
        if next_token not in curr_dict:
            minimum = 0
            is_leaf = 0
            depth = 0
        else:
            dkey = (next_token, "descendant")
            # Added a universal minimum probability for every edge of the tree. 
            # Explanation: When initial probabilities are too small, multiplicative updates are inconsequential
            minimum = np.minimum(1.0, np.maximum(curr_dict[dkey] * multiplier / (anc_prob + EPS), MIN_PROB))

            anc2 = anc+[next_token]
            # inp2 = pad_sequence(anc2)
            inp2 = sequence_to_key(anc2, padding=True, empty=empty, length=seq_len)
            is_leaf = int(inp2 not in tree)

            # depth = jnp.amin(curr_dict["depths"])
            depth = min(curr_dict["depths"])
            
        leaves.append(is_leaf)
        minimums.append(minimum)
        depths.append(depth)
    target_probs = get_target_probs(minimums, leaves, depths, multiplier, prob_dict[inp])
    result = {inp:target_probs}

    # recursively calculate targets for children
    for next_token in range(token_num):
        if next_token in curr_dict:
            prob = curr_dict[next_token]
            target_prob = target_probs[next_token]
            anc2 = anc+[next_token]
            anc_prob2 = anc_prob * target_prob
            multiplier2 = multiplier / (target_prob / (prob + EPS) + EPS)
            curr_result = build_update_dict(anc2, anc_prob2, multiplier2, tree, prob_dict, token_num, empty, seq_len)
            # result = result | curr_result
            result.update(curr_result)
    return result




def seq_prp_targets(sequences, probs, token_num, global_multiplier, empty=0, verbose=False):
    prob_dict, seq_len = get_prob_dict(sequences, probs, empty=empty)
    tree = build_tree(sequences, prob_dict)

    # print("Tree:")
    # for k in tree.keys():
    #     print(k, " -> ", tree[k])

    # print("Prob:")
    # for k in prob_dict.keys():
    #     print(k, " -> Max ", np.max(np.around(prob_dict[k], 2)))

    init_prob = 1.0
    root = []
    update_dict = build_update_dict(root, init_prob, global_multiplier, tree, prob_dict, token_num, empty, seq_len)

    if verbose:
        print("Targets:")
        for k in update_dict.keys():
            # print(k, " -> ", np.around(update_dict[k], 2))
            print(k, " -> ", update_dict[k])

    # create target matrix
    targets = copy.deepcopy(probs)
    for s_idx, s in enumerate(sequences):
        seq_len = len(s)
        for i in range(seq_len):
            next_token = s[i]
            prefix = list(s)[:i]
            sequence_key = sequence_to_key(prefix, padding=True, empty=empty, length=seq_len)
            targets[s_idx][i][next_token] = update_dict[sequence_key][next_token]
    # targets = jnp.array(targets)
    return targets



# TODO: use empty parameter when calling!
def get_prob_dict(sequences, probs, empty):
    prob_dict = {}
    max_seq_len = max([len(s) for s in sequences])
    terminal_probs = copy.deepcopy(probs[0][0])
    terminal_probs.fill(0.0)
    terminal_probs[empty] = 1.0
    for seq, prob in zip(sequences, probs):
        for s in range(len(seq)+1):
            prefix = list(seq)[:s] # TODO: why does it need to be a list instead of ndarray?
            prefix_key = sequence_to_key(prefix, padding=True, empty=empty, length=max_seq_len)
            if s < len(seq):
                prob_dict[prefix_key] = prob[s]
            else:
                prob_dict[prefix_key] = terminal_probs
    return prob_dict, max_seq_len


def get_prob_weights(sequences, empty, shape, dtype):
    # sequences shape (support * bs * seq)
    # weights shape (support * bs * seq * tokens)
    weights = np.zeros(shape, dtype=dtype)
    # Fix b and s the batch (this is the item in the batch) and seq (this is the position) dimensions: for all s in the support, if token == to token init, do weight += 1.0
    for index, token in np.ndenumerate(sequences):
        support = sequences[:,index[1],index[2]]
        if token!=empty:
            for s,token_s in enumerate(support):
                if token_s==token:
                    weights[s,index[1],index[2],token] += 1.0

    weights = np.reciprocal(weights, where= weights!=0.0)
    weights[weights<EPS] = 0
    return weights




