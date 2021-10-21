import tensorflow as tf
from tensorflow import keras
from functools import partial
import time
import numpy as np

import data

tf.random.set_seed(13)

                            

class EuclideanMetric:
    def __init__(self):
        pass

    def dist(self, emb1, emb2):
        return tf.sqrt(tf.reduce_sum(tf.square(emb1 - emb2)) / emb1.shape[0])

    def score(self, emb1, emb2): # a value between 0 and 1
        d = self.dist(emb1, emb2)
        return tf.exp(-d)

    def dist2loss(self, dist, positive):
        if positive:
            return dist
        else:
            return - dist


class CosineMetric:
    def __init__(self):
        self.cosineSimilarity = keras.losses.CosineSimilarity()

    def score(self, emb1, emb2):
        return tf.abs(self.cosineSimilarity(emb1, emb2))

    def dist(self, emb1, emb2):
        s = self.score(emb1, emb2)
        return 1 - s

    def dist2loss(self, dist, positive):
        if positive:
            return dist
        else:
            return 1 - dist

myMetric = EuclideanMetric()

    

def mapping2score(mapping, emap):
    score = tf.constant(1.0)
    vars = []
    for aux in mapping:
        pred, values = mapping[aux]
        if len(pred) == 1: # auxiliary constant
            emb_aux = emap.embedding(aux, "aconst")
            emb_source = emap.embedding(values[0], "sconst")
            # vars.append(emb_source)
            score *= myMetric.score(emb_aux, emb_source)
        elif len(pred) == 2: # unary predicate
            emb_aux = emap.embedding(aux, "apred")
            pred_name = "{}_{}".format(pred[0], pred[1])
            emb_pred = emap.embedding(pred_name, "spred")
            # vars.append(emb_pred)
            score *= myMetric.score(emb_aux, emb_pred)
        elif len(pred) == 3: # binary predicate
            emb_aux = emap.embedding(aux, "apred")
            pred_name = "{}_{}_{}".format(pred[0], pred[1], pred[2])
            emb_pred = emap.embedding(pred_name, "spred")
            # vars.append(emb_pred)
            score *= myMetric.score(emb_aux, emb_pred)
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
        self.slack_variables_list = []
        for mappings in mappings_list:
            if len(mappings) > 0:
                curr_emb_aux_list, curr_emb_source_list, slack_variables_list = self.process_mappings(mappings)
                self.emb_aux_list.append(curr_emb_aux_list)
                self.emb_source_list.append(curr_emb_source_list)
                self.slack_variables_list.append(slack_variables_list)
        assert len(self.emb_source_list) > 0, "TODO"

    def process_mappings(self, mappings):
        assert len(mappings) > 0
        aux_names = mappings[0].keys()        
        emb_aux_list = []
        emb_source_list = [[] for _ in range(len(aux_names))]
        slack_variables_list = [[] for _ in range(len(aux_names))]
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
                slack_variables_list[i].append(tf.Variable(tf.random.uniform([], minval=0, maxval=1)))
        return emb_aux_list, emb_source_list, slack_variables_list


    def call(self, Temp):
        distances_per_rule = []
        variables = []
        all_slack_variables = []
        slack_loss = tf.constant(0.0)
        for curr_emb_aux_list, curr_emb_source_list, curr_slack_variables_list in zip(self.emb_aux_list, self.emb_source_list, self.slack_variables_list):
            variables += curr_emb_aux_list
            distances_per_pred = []
            for emb_aux, emb_source, slack_variables in zip(curr_emb_aux_list, curr_emb_source_list, curr_slack_variables_list):
                all_slack_variables += slack_variables
                distances = []
                slack_point = tf.constant(0.0)
                slack_sum = tf.constant(0.0)
                for emb2, slack in zip(emb_source, slack_variables):
                    slack_loss += tf.maximum(1.0, slack) - 1 - tf.minimum(0.0, slack)
                    slack_point += slack * emb2
                    slack_sum += slack
                    distances.append(myMetric.dist(emb_aux, emb2))
                if self.positive:
                    slack_loss += tf.square(slack_sum - 1.0)
                    distances_per_pred.append(myMetric.dist(emb_aux, slack_point))
                else:
                    distances_per_pred.append(distances)
            if not self.positive:
                distances_per_pred = tf.reduce_mean(distances_per_pred, axis=0)
            
            # score_per_rule1 = tf.reduce_sum(tf.square(score_per_pred) * softmax(score_per_pred, Temp))
            distance_per_rule = tf.reduce_max(distances_per_pred)
            distances_per_rule.append(distance_per_rule)

        distance = tf.reduce_min(distances_per_rule)
        loss1 = myMetric.dist2loss(distance, self.positive)
        if self.positive:
            loss2 = slack_loss
            variables += slack_variables
        else:
            loss2 = tf.constant(0.0)
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

        # print(self.fact, self.positive, " loss: ", loss)
        return loss
        

def train(generator, epochs, Tmax=10.0, Tmin=0.01, lr=0.001, neg_coeff = 1.0, reg_coeff = 0.0):
    # Temp is going to be an exponentially diminishing curve that fits to Tmax and Tmin
    # Temp = alpha * exp(-beta * (epoch+1))
    beta = np.log(Tmax / Tmin) / (epochs-1)
    alpha = Tmax / np.exp(-beta)

    optimizer = tf.keras.optimizers.SGD(learning_rate=lr, momentum=0.9)
    # optimizer = tf.keras.optimizers.Adam(learning_rate=lr)
    t0 = time.time()    
    for epoch in range(epochs):
        T = alpha * np.exp(-beta * (epoch+1))

        emb1 = generator.emap.embedding("Author_pred1a", "aux")
        emb2 = generator.emap.embedding("authors_id", "source")

        aux_list, pos_list, neg_list = generator.get_all()
        reg_embeddings = generator.get_embedding_sample(size=32)
        batch_loss = 0
        for aux, pos, neg in zip(aux_list, pos_list, neg_list):
            vars = [aux] + pos + neg + reg_embeddings
            with tf.GradientTape() as g:
                reg_dist = data.squared_distance(tf.convert_to_tensor(reg_embeddings))
                reg_loss = -tf.reduce_sum(reg_dist)
                loss = tf.constant(0.0)
                pos_dist = []
                neg_dist = []
                for p in pos:
                    pos_dist.append(myMetric.dist(p, aux))
                for n in neg:
                    neg_dist.append(myMetric.dist(n, aux))
                pos_loss = tf.reduce_min(pos_dist)
                neg_loss = - tf.reduce_min(neg_dist)
                loss = pos_loss + neg_coeff * neg_loss + reg_coeff * reg_loss
                loss1 = loss + tf.reduce_sum(tf.square(emb1-emb2))
                batch_loss += loss.numpy()
            grads=g.gradient(loss, [aux])
            optimizer.apply_gradients(zip(grads, vars))
        print("{} Batch loss: {}".format(epoch, batch_loss / generator.batch_size))
        generator.emap.find_closest("aux","source", myMetric.score)
        
        # indices = np.random.permutation(len(models))
        # epoch_loss1 = 0
        # epoch_loss2 = 0
        # batch_grads = []
        # batch_vars = []
        # for i in range(len(models)):
        #     index = indices[i]
        #     m = models[index]
        #     with tf.GradientTape() as g:
        #         loss1, loss2, vars = m(T)
        #         epoch_loss1 += loss1.numpy()
        #         epoch_loss2 += loss2.numpy()
        #         loss = loss1 + loss2
        #     grads = g.gradient(loss, vars)
        #     batch_grads += grads
        #     batch_vars += vars
        #     if (i+1) % batch_size == 0:
        #         optimizer.apply_gradients(zip(batch_grads, batch_vars))
        #         batch_grads = []
        #         batch_vars = []
        # t1 = time.time()
        # epoch_loss1 /= len(models)
        # epoch_loss2 /= len(models)
        # print("Epoch {}, loss {} ({}+{}), temp {}, time {} sec".format(epoch, epoch_loss1 + epoch_loss2, epoch_loss1, epoch_loss2, T, t1-t0))
        # emap.find_closest("aconst","sconst")
        # emap.find_closest("apred", "spred")

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

