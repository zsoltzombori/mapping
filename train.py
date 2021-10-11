import tensorflow as tf
from tensorflow import keras
from functools import partial
import time

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
            closest_dist = 1e10
            for n2 in d2:
                emb2 = d2[n2]
                curr_dist = dist(emb1, emb2)
                if curr_dist < closest_dist:
                    closest_dist = curr_dist
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


def dist(emb1, emb2):
    return tf.sqrt(tf.reduce_sum(tf.square(emb1 - emb2)))

def mapping2loss(mapping, emap):
    loss = tf.constant(0.0)
    vars = []
    for aux in mapping:
        pred, values = mapping[aux]
        if len(pred) == 1: # auxiliary constant
            emb_aux = emap.embedding(aux, "aconst")
            emb_source = emap.embedding(values[0], "sconst")
            vars.append(emb_source)
            loss += dist(emb_aux, emb_source)
        elif len(pred) == 2: # unary predicate
            emb_aux = emap.embedding(aux, "apred")
            pred_name = "{}_{}".format(pred[0], pred[1])
            emb_pred = emap.embedding(pred_name, "spred")
            vars.append(emb_pred)
            loss += dist(emb_aux, emb_pred)
        elif len(pred) == 3: # binary predicate
            emb_aux = emap.embedding(aux, "apred")
            pred_name = "{}_{}_{}".format(pred[0], pred[1], pred[2])
            emb_pred = emap.embedding(pred_name, "spred")
            vars.append(emb_pred)
            loss += dist(emb_aux, emb_pred)
        vars.append(emb_aux)
    return loss, vars

def softmin(tensor, Temp):
    e = 0.1
    tensor2 = tensor/(Temp+e)
    exps = tf.exp(tensor2 - tf.reduce_min(tensor2))
    return exps / tf.reduce_sum(exps, axis=0)

def reduce_softmin(tensor, Temp):
    sm = softmin(tensor, Temp)
    return tf.reduce_sum(tensor * sm)

def rule2loss(mappings, emap, Temp):
    losses = []
    vars = []
    for m in mappings:
        l, v = mapping2loss(m, emap)
        losses.append(l)
        vars += v
    losses = tf.convert_to_tensor(losses)

    sm = softmin(losses, Temp)
    loss = tf.reduce_sum(losses * sm)
    index = tf.argmin(losses)
    return loss, vars, index

def fact2loss(mappings_list, emap, Temp):
    losses = []
    vars = []
    indices = []
    for m in mappings_list:
        l, v, i = rule2loss(m, emap, Temp)
        losses.append(l)
        vars += v
        indices.append(i)
    losses = tf.convert_to_tensor(losses)
    loss = tf.reduce_min(losses)
    index = tf.argmin(losses)
    return loss, vars #, (index, indices[index])

class MappingFactory():
    def __init__(self, emap):
        self.emap = emap

    def create(self, mappings_list):
        Temp = keras.Input(shape=(1, ))
        loss, vars = fact2loss(mappings_list, self.emap, Temp)
        model = keras.Model(inputs=Temp, outputs = (loss, vars))
        return model

class MappingLayer(keras.layers.Layer):
    def __init__(self, mappings_list, emap):
        super(MappingLayer, self).__init__()
        self.emap = emap
        self.mappings_list = mappings_list
        # self.vars = []
        # loss_list = []
        # row_splits = [0]
        # for mappings in mappings_list:
        #     for m in mappings:
        #         loss, vars = mapping2loss(m, emap)
        #         loss_list.append(loss)
        #         self.vars += vars
        #     row_splits.append(len(loss_list))
        # self.loss_tensor = tf.RaggedTensor.from_row_splits(values=loss_list, row_splits=row_splits)

    def call(self, Temp):
        var_list = []
        loss_list = []
        row_splits = [0]
        for mappings in self.mappings_list:
            for m in mappings:
                loss, vars = mapping2loss(m, self.emap)
                loss_list.append(loss)
                var_list += vars
            row_splits.append(len(loss_list))
        self.loss_tensor = tf.RaggedTensor.from_row_splits(values=loss_list, row_splits=row_splits)
        reduce_partial = partial(reduce_softmin, Temp=Temp)
        losses = tf.map_fn(reduce_partial, self.loss_tensor, fn_output_signature=tf.float32)
        loss = tf.reduce_min(losses)
        return loss, var_list

# emap = EmbeddingMap(4)
# model = MappingLayer([mappings1, mappings2], emap)

def train(layers, epochs, emap, T):
    optimizer = tf.keras.optimizers.SGD(learning_rate=1e-3)
    for i in range(epochs):
        t0 = time.time()
        total_loss = 0
        for l in layers:
            with tf.GradientTape() as g:
                loss, vars = l(T)
                total_loss += loss.numpy()
            grads = g.gradient(loss, vars)
            optimizer.apply_gradients(zip(grads, vars))
        t1 = time.time()
        print("Epoch {}, loss {}, time {} sec".format(i, loss, t1-t0))
        # emap.print()
        emap.find_closest("aconst","sconst")
        emap.find_closest("apred", "spred")

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

