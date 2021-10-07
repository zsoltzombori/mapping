import tensorflow as tf
from tensorflow import keras

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
    e = 1.0e-1
    return tf.exp(-tensor/(Temp+e)) / tf.reduce_sum(tf.exp(-tensor/(Temp+e)), axis=0)

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
    return loss, vars, (index, indices[index])

emap = EmbeddingMap(4)
optimizer = tf.keras.optimizers.SGD(learning_rate=1e-3)


for i in range(1000):
    with tf.GradientTape() as g:
        loss, vars, index = fact2loss([mappings1, mappings2], emap, 2/(i+1))
        print(i, "loss: ", loss)
        # print("index: ", index)
        if i % 100 == 0:
            emap.print()
            emap.find_closest("aconst","sconst")
            emap.find_closest("apred", "spred")
        grads = g.gradient(loss, vars)
        optimizer.apply_gradients(zip(grads, vars))

emap.find_closest("aconst","sconst")
emap.find_closest("apred", "spred")

class MappingLayer(keras.layers.Layer):
    def __init__(self, mappings_list, emap, Temp):
        super(MappingLayer, self).__init__()
        self.loss, self.vars, self.index = fact2loss(mappings_list, emap, Temp)

    def call(self, inputs):
        pass

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

