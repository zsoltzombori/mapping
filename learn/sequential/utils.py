import os
import sys
import copy
import jax
import jax.numpy as jnp


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
            ratios.append(seq_prob_list[i] / seq_prob_list[j])
    # print("Ratios: ", np.around(np.asarray(ratios),2))
    # print("Sum prob: ", jnp.sum(jnp.array(seq_prob_list)))
    return tree



def get_target_probs(minimums, leaves, depths, multiplier, probs):    
    minimums = jnp.array(minimums)
    # print("Minimums", minimums)
    leaves = jnp.array(leaves)
    depths = jnp.array(depths)

    slacks = jnp.equal(minimums, 0)
    nonleaves = jnp.equal(leaves, 0)

    # leaves should get their minimum
    leaf_targets = leaves * minimums

    # nonleaf targets start from the minimum and then get promoted
    nonleaf_targets = nonleaves * minimums

    capacity = 1 - jnp.sum(minimums)

    curr_multiplier = multiplier ** (1/jnp.maximum(1,depths))
    ideal_nonleaf_targets = (slacks * probs + (1-slacks) * curr_multiplier * probs) * nonleaves

    direction = jnp.maximum(0.0, ideal_nonleaf_targets - nonleaf_targets)
    if jnp.sum(direction) > 0:
        nonleaf_targets += direction / jnp.sum(direction) * capacity

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
            minimum = jnp.minimum(1.0, curr_dict[dkey] * multiplier / anc_prob)

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
            multiplier2 = multiplier / (target_prob / prob)
            curr_result = build_update_dict(anc2, anc_prob2, multiplier2, tree, prob_dict, token_num, empty, seq_len)
            # result = result | curr_result
            result.update(curr_result)
    return result




def seq_prp_targets(sequences, probs, token_num, global_multiplier, empty=0):
    prob_dict, seq_len = get_prob_dict(sequences, probs, empty=empty)
    tree = build_tree(sequences, prob_dict)
    # print("Tree:")
    # for k in tree.keys():
    #     print(k, " -> ", tree[k])

    # print("Prob:")
    # for k in prob_dict.keys():
    #     print(k, " -> ", jnp.around(prob_dict[k], 2))

    init_prob = 1.0
    root = []
    update_dict = build_update_dict(root, init_prob, global_multiplier, tree, prob_dict, token_num, empty, seq_len)

    # print("Targets:")
    # for k in update_dict.keys():
    #     print(k, " -> ", jnp.around(update_dict[k], 2))

    # create target matrix
    targets = []
    for seq in sequences:
        # sequence_key = sequence_to_key(seq, padding=True, empty=empty, length=seq_len)
        sequence_key = sequence_to_key(seq) # sequences are already max length
        targets.append(update_dict[sequence_key])
    targets = jnp.array(targets)

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
