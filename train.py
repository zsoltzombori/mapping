import tensorflow as tf
from tensorflow import keras
from functools import partial
import time
import numpy as np

tf.random.set_seed(20)

class EmbeddingMap:
    def __init__(self, dim):
        self.dim = dim
        self.emb_dict = {}

    def create_embedding(self):
        return tf.Variable(tf.random.normal((self.dim, )))
        
    def embedding(self, name, entity_type):
        if entity_type not in self.emb_dict:
            self.emb_dict[entity_type] = {}
            
        d = self.emb_dict[entity_type]
        if name in d:
            return d[name]
        else:
            v = self.create_embedding()
            d[name] = v
            return v

    def print(self):
        print("Embeddings:")
        for k in sorted(self.emb_dict.keys()):
            d = self.emb_dict[k]
            print("   {}   ".format(k))
            for n in d:
                print("\t{}\t{}".format(n, d[n].value()))

    def find_closest(self, type1, type2):
        if type1 not in self.emb_dict or type2 not in self.emb_dict:
            return

        d1 = self.emb_dict[type1]
        d2 = self.emb_dict[type2]
        for n1 in d1:
            emb1 = d1[n1]
            closest = None
            closest_score = -100000
            for n2 in d2:
                emb2 = d2[n2]
                curr_score = similarity(emb1, emb2)
                if curr_score > closest_score:
                    closest_score = curr_score
                    closest = n2
            print("{} is closest to {}".format(n1, closest))
                            

mappings1 = [
    {'a1': (('Paper', 'paperID', 'TYPE'), (179, 1)), 'b1': (('Person', 'is_Reviewer', 'TYPE'), (False, 1))},
    {'a1': (('Person', 'ID', 'is_Author'), (179, True)), 'b1': (('Person', 'is_user', 'is_Reviewer'), (True, True))},
]

mappings2 = [
    {'a2': (('Paper', 'hasAuthor'), (179,))},
    {'a2': (('Paper', 'paperID'), (179,))},
    {'a2': (('Person', 'ID'), (179,))},
]


cosineSimilarity = keras.losses.CosineSimilarity()

# @tf.function
def similarity(emb1, emb2):
    # return -tf.sqrt(tf.reduce_sum(tf.square(emb1 - emb2)) / emb1.shape[0])
    return tf.abs(cosineSimilarity(emb1, emb2))

def mapping2score(mapping, emap):
    score = tf.constant(1.0)
    vars = []
    for aux in mapping:
        pred, values = mapping[aux]
        if len(pred) == 1: # auxiliary constant
            emb_aux = emap.embedding(aux, "aconst")
            emb_source = emap.embedding(values[0], "sconst")
            # vars.append(emb_source)
            score *= similarity(emb_aux, emb_source)
        elif len(pred) == 2: # unary predicate
            emb_aux = emap.embedding(aux, "apred")
            pred_name = "{}_{}".format(pred[0], pred[1])
            emb_pred = emap.embedding(pred_name, "spred")
            # vars.append(emb_pred)
            score *= similarity(emb_aux, emb_pred)
        elif len(pred) == 3: # binary predicate
            emb_aux = emap.embedding(aux, "apred")
            pred_name = "{}_{}_{}".format(pred[0], pred[1], pred[2])
            emb_pred = emap.embedding(pred_name, "spred")
            # vars.append(emb_pred)
            score *= similarity(emb_aux, emb_pred)
        vars.append(emb_aux)
    return score, vars

# def mapping2pairs(mapping, emap):
#     pairs = []
#     var_names = []
#     for aux in mapping:
#         pred, values = mapping[aux]
#         if len(pred) == 1: # auxiliary constant
#             emb_aux = emap.embedding(aux, "aconst")
#             emb_source = emap.embedding(values[0], "sconst")
#             var_names.append((aux, "aconst"))
#         elif len(pred) == 2: # unary predicate
#             emb_aux = emap.embedding(aux, "apred")
#             pred_name = "{}_{}".format(pred[0], pred[1])
#             emb_source = emap.embedding(pred_name, "spred")
#             var_names.append((aux, "spred"))
#         elif len(pred) == 3: # binary predicate
#             emb_aux = emap.embedding(aux, "apred")
#             pred_name = "{}_{}_{}".format(pred[0], pred[1], pred[2])
#             emb_source = emap.embedding(pred_name, "spred")
#             var_names.append((aux, "spred"))
#         pairs.append([emb_aux, emb_source])
#     pairs = tf.convert_to_tensor(pairs)
#     return pairs, var_names


def softmin(tensor, Temp):
    e = 0.1
    tensor2 = tensor/(Temp+e)
    exps = tf.exp(-tensor2 - tf.reduce_min(tensor2))
    return exps / tf.reduce_sum(exps, axis=0)
def softmax(tensor, Temp):
    e = 0.1
    tensor2 = tensor/(Temp+e)
    exps = tf.exp(tensor2 - tf.reduce_max(tensor2))
    return exps / tf.reduce_sum(exps, axis=0)

# def reduce_softmin(tensor, Temp):
#     sm = softmin(tensor, Temp)
#     return tf.reduce_sum(tensor * sm)
# def reduce_softmax(tensor, Temp):
#     sm = softmax(tensor, Temp)
#     return tf.reduce_sum(tensor * sm)

# def rule2score(mappings, emap, Temp):
#     scores = []
#     vars = []
#     for m in mappings:
#         s, v = mapping2score(m, emap)
#         scores.append(l)
#         vars += v
#     index = tf.argmax(scores)
#     sm = softmax(scores, Temp)
#     score = tf.reduce_sum(scores * sm)
#     return score, vars, index

# def fact2score(mappings_list, emap, Temp):
#     scores = []
#     vars = []
#     indices = []
#     for m in mappings_list:
#         s, v, i = rule2score(m, emap, Temp)
#         scores.append(l)
#         vars += v
#         indices.append(i)
#     scores = tf.convert_to_tensor(scores)
#     score = tf.reduce_max(scores)
#     index = tf.argmax(scores)
#     return score, vars, (index, indices[index])

class MappingLayer(keras.layers.Layer):
    def __init__(self, mappings_list, emap, fact, positive=True):
        super(MappingLayer, self).__init__()
        self.emap = emap
        self.mappings_list = mappings_list
        self.fact = fact
        self.positive = positive

        self.emb_aux_list = []
        self.emb_source_list = []
        for mappings in mappings_list:
            if len(mappings) > 0:
                curr_emb_aux_list, curr_emb_source_list = self.process_mappings(mappings)
                self.emb_aux_list.append(curr_emb_aux_list)
                self.emb_source_list.append(curr_emb_source_list)

    def process_mappings(self, mappings):
        assert len(mappings) > 0
        aux_names = mappings[0].keys()        
        emb_aux_list = []
        emb_source_list = [[] for _ in range(len(aux_names))]
        first_row = True
        for m in mappings:
            for i, aux_name in enumerate(aux_names):
                pred, values = m[aux_name]
                if len(pred) == 1:
                    atype = "aconst"
                    emb_source = self.emap.embedding(values[0], "sconst")
                elif len(pred) == 2:
                    atype = "apred"
                    pred_name = "{}_{}".format(pred[0], pred[1])
                    emb_source = self.emap.embedding(pred_name, "spred")
                elif len(pred) == 3:
                    atype = "apred"
                    pred_name = "{}_{}_{}".format(pred[0], pred[1], pred[2])
                    emb_source = self.emap.embedding(pred_name, "spred")
                if first_row:
                    emb_aux = self.emap.embedding(aux_name, atype)
                    emb_aux_list.append(emb_aux)
                emb_source_list[i].append(emb_source)
        return emb_aux_list, emb_source_list


    def call(self, Temp):
        scores_per_rule1 = []
        scores_per_rule2 = []
        variables = []
        source_loss = tf.constant(0.0)
        for curr_emb_aux_list, curr_emb_source_list in zip(self.emb_aux_list, self.emb_source_list):
            variables += curr_emb_aux_list
            scores_per_pred = []
            for emb_aux, emb_source in zip(curr_emb_aux_list, curr_emb_source_list):
                for i, emb1 in enumerate(emb_source):
                    variables.append(emb1)
                    for j, emb2 in enumerate(emb_source):
                        if j > i:
                            variables.append(emb2)
                            source_loss += similarity(emb1, emb2)
                # variables += emb_source
                partial_similarity = partial(similarity, emb2 = emb_aux)
                scores = tf.map_fn(partial_similarity, tf.convert_to_tensor(emb_source), fn_output_signature=tf.float32)
                scores_per_pred.append(scores)
            score_per_pred = tf.reduce_prod(scores_per_pred, axis=0)
            # score_per_rule1 = tf.reduce_sum(tf.square(score_per_pred) * softmax(score_per_pred, Temp))
            score_per_rule1 = tf.reduce_max(score_per_pred)
            score_per_rule2 = tf.reduce_max(score_per_pred)
            scores_per_rule1.append(score_per_rule1)
            scores_per_rule2.append(score_per_rule2)
        if len(scores_per_rule1) == 0:
            score1 = tf.constant(0.0)
        else:
            score1 = tf.reduce_max(scores_per_rule1)
        if len(scores_per_rule2) == 0:
            score2 = tf.constant(0.0)
        else:
            score2 = tf.reduce_max(scores_per_rule2)
        
        if self.positive:
            loss1 = 1-score1
            loss2 = 1-score2
        else:
            loss1 = score1
            loss2 = score2
        loss1 = tf.pow(loss1, 10)
        loss1 += source_loss
        return loss1, loss2, variables


    def eval(self):
        score_list = []
        for mappings in self.mappings_list:
            for m in mappings:
                score, _vars = mapping2score(m, self.emap)
                score_list.append(score.numpy())
        if len(score_list) == 0:
            score = 0.0
        else:
            score = np.max(score_list)
        if self.positive:
            loss = 1-score
        else:
            loss = score

        print(self.fact, self.positive, " loss: ", loss)
        return loss
        

def train(models, epochs, emap, batch_size=32, Tmax=10.0, Tmin=0.01, lr=0.001):
    print("Datapoints: ", len(models))
    assert len(models) > batch_size, "Batch size ({}) should be smaller than number of datapoints ({})".format(batch_size, len(models))

    # Temp is going to be an exponentially diminishing curve that fits to Tmax and Tmin
    # Temp = alpha * exp(-beta * (epoch+1))
    beta = np.log(Tmax / Tmin) / (epochs-1)
    alpha = Tmax / np.exp(-beta)

    # optimizer = tf.keras.optimizers.SGD(learning_rate=lr)
    optimizer = tf.keras.optimizers.Adam(learning_rate=lr)
    t0 = time.time()    
    for epoch in range(epochs):
        T = alpha * np.exp(-beta * (epoch+1))
        indices = np.random.permutation(len(models))
        epoch_loss1 = 0
        epoch_loss2 = 0
        batch_grads = []
        batch_vars = []
        for i in range(len(models)):
            index = indices[i]
            m = models[index]
            with tf.GradientTape() as g:
                loss1, loss2, vars = m(T)
                epoch_loss1 += loss1.numpy()
                epoch_loss2 += loss2.numpy()
            grads = g.gradient(loss1, vars)
            batch_grads += grads
            batch_vars += vars
            if (i+1) % batch_size == 0:
                optimizer.apply_gradients(zip(batch_grads, batch_vars))
                batch_grads = []
                batch_vars = []
        t1 = time.time()
        print("Epoch {}, loss {}-{}, temp {}, time {} sec".format(epoch, epoch_loss1 / len(models), epoch_loss2 / len(models), T, t1-t0))
        emap.find_closest("aconst","sconst")
        emap.find_closest("apred", "spred")

def eval(models):
    loss = 0
    for m in models:
        loss += m.eval()
    return loss / len(models)
        
# emap.find_closest("aconst","sconst")
# emap.find_closest("apred", "spred")


# mapping rules
# C(X):- A(X).
# C(X):- A(X,Constant(Z)).
# C(X):- A(X,Y).
# C(X):- A(X,Y), B(Y).

# R(X,Y):- A(X,Y).
# R(X,Y):- A(X), B(X,Y).
# R(X,Y):- A(X), B(X,Y), C(Y).
# R(X,Y):- A(X,Y), B(X,Constant(Z)).
# R(X,Y):- A(X,Y), B(Y,Constant(Z)).

