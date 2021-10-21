import tensorflow as tf
from tensorflow import keras
import random

class EmbeddingMap:
    def __init__(self, dim):
        self.dim = dim
        self.emb_dict = {}

    def create_embedding(self):
        # return tf.Variable(tf.random.normal((self.dim, )))
        return tf.Variable(tf.random.uniform((self.dim,), minval=0, maxval=1))
        
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

    def find_closest(self, type1, type2, score):
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
                curr_score = score(emb1, emb2)
                if curr_score > closest_score:
                    closest_score = curr_score
                    closest = n2
            print("{} is closest to {}".format(n1, closest))

def squared_distance(A):
    r = tf.reduce_sum(A*A, 1)
    # turn r into column vector
    r = tf.reshape(r, [-1, 1])
    D = r - 2*tf.matmul(A, tf.transpose(A)) + tf.transpose(r)
    return D

class MyGenerator(keras.utils.Sequence):

    def __init__(self, dim=32, batch_size=32):
        self.dim = dim
        self.batch_size = batch_size
        self.emap = EmbeddingMap(dim)
        self.positive_mappings = {}
        self.negative_mappings = {}

    def add_mappings(self, mappings, positive):
        dict0 = self.process_mappings(mappings)
        if positive:
            self.add_value_to_dict(self.positive_mappings, dict0)
        else:
            self.add_value_to_dict(self.negative_mappings, dict0)

    def add_value_to_dict(self, dict_of_list, dict0):
        for aux in dict0:
            if aux not in dict_of_list:
                dict_of_list[aux] = []
            dict_of_list[aux].append(dict0[aux])
            
    # returns list of [aux, sources] pairs
    # aux is an auxiliary predicate or constant
    # sources is a list of source predicates or constants
    # aux can be mapped to any symbol in sources
    def process_mappings(self, mappings):
        result = {}
        for m in mappings:
            for aux in m:
                pred, values = m[aux]
                if len(pred) == 1:
                    source = values[0]
                else:
                    source = "_".join(pred)
                self.emap.embedding(aux, "aux")
                self.emap.embedding(source, "source")

                if aux not in result:
                    result[aux] = []
                result[aux].append(source)
        return result

    def get_batch(self):
        aux_set = set(self.positive_mappings.keys()).union(set(self.negative_mappings.keys()))

        aux_list = []
        positives_list = []
        negatives_list = []

        aux_batch = random.choices(list(aux_set), k=self.batch_size)
        for aux in aux_batch:
            if aux not in self.positive_mappings:
                positives = []
            else:
                positives = random.sample(self.positive_mappings[aux], k=1)[0]
            if aux not in self.negative_mappings:
                negatives = []
            else:
                negatives = random.sample(self.negative_mappings[aux], k=1)[0]

            positives = list(set(positives) - set(negatives)) # TODO RECONSIDER

            positives = [self.emap.embedding(p, "source") for p in positives]
            negatives = [self.emap.embedding(n, "source") for n in negatives]
            positives_list.append(positives)
            negatives_list.append(negatives)
            aux_list.append(self.emap.embedding(aux, "aux"))

        return aux_list, positives_list, negatives_list

    def get_all(self):
        aux_set = set(self.positive_mappings.keys()).union(set(self.negative_mappings.keys()))

        aux_list = []
        positives_list = []
        negatives_list = []

        for aux in list(aux_set):
            if aux not in self.positive_mappings:
                positives_l = [[]]
            else:
                positives_l = self.positive_mappings[aux]
            if aux not in self.negative_mappings:
                negatives_l = [[]]
            else:
                negatives_l = self.negative_mappings[aux]

            for positives0 in positives_l:
                for negatives0 in negatives_l:
                    
                    positives = list(set(positives0) - set(negatives0)) # TODO RECONSIDER
                    negatives = negatives0

                    positives = [self.emap.embedding(p, "source") for p in positives]
                    negatives = [self.emap.embedding(n, "source") for n in negatives]
                    positives_list.append(positives)
                    negatives_list.append(negatives)
                    aux_list.append(self.emap.embedding(aux, "aux"))
        return aux_list, positives_list, negatives_list


    # random sample of embeddings
    def get_embedding_sample(self, size, embtype="source"):
        d = self.emap.emb_dict[embtype]
        keys = list(d.keys())
        if len(keys) > size:
            keys = random.sample(keys, size)
        return [d[k] for k in keys]

        
