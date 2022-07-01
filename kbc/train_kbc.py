import numpy as np
import tensorflow as tf
import time
import sys

from monitor import MonitorProbs
import pickle

LR=0.1
LOSS_TYPE="nll" #nll, prp, prp2
MONITOR=False

EPS=1e-5
logEPS=tf.math.log(EPS)

class Tokenizer():
    def __init__(self):
        self.vocab_in = []
        self.vocab_out = []

    def input2index(self, seq):
        (head, pred, tail) = seq
        key = pred
        if key not in self.vocab_in:
            self.vocab_in.append(key)
        return self.vocab_in.index(key)

    def output2index(self, seq):
        key = "|||".join(seq)
        if key not in self.vocab_out:
            self.vocab_out.append(key)
        return self.vocab_out.index(key)
    

    def tokenize_dict(self, in_dict):
        result = []
        for inp in in_dict:
            outputs = in_dict[inp]
            if len(outputs) > 0:
                inp_key = self.input2index(inp)
                out_keys = [self.output2index(out) for out in outputs]
                result.append((inp_key, out_keys))
        return result
                
    

def prepare_data(data, num_classes):    
    inputs = []
    outputs = []
    for d in data:
        inputs.append(float(d[0]))
        o = [int(x in d[1]) for x in range(num_classes)]
        outputs.append(o)
    return inputs, outputs


    
def make_batches(inputs, outputs, batch_size):
    l = len(inputs)
    assert len(outputs) == l
    perm = np.random.permutation(l)
    for ndx in range(0, l, batch_size):
        indices = perm[ndx:min(ndx+batch_size,l)]
        i = [inputs[x] for x in indices]
        o = [outputs[x] for x in indices]
        i = np.array(i)
        i = np.expand_dims(i, axis=1)
        o = np.array(o)
        yield (i, o)

def linear_loss(pred, real, ispositive, which="both"):
    # we are assuming that pred is the logit vector not the prob vector!!!
    logits_up = real * pred
    logits_down = (1-real) * pred

    probs = tf.nn.softmax(pred) * real
    sumprobs = tf.reduce_sum(probs, axis=-1)
    sumprobs = tf.reduce_mean(sumprobs)
    
    if which == "both":
        loss = tf.reduce_sum(logits_down - logits_up, axis=-1)
    elif which == "up":
        loss = tf.reduce_sum(- logits_up, axis=-1)
    elif which == "down":
        loss = tf.reduce_sum(logits_down, axis=-1)

    if not ispositive:
        loss *= -1
    return loss, sumprobs, probs

def nll_loss(pred, real, ispositive):
    probs = real * pred
    sumprobs = tf.reduce_sum(probs, axis=-1)
    sumprobs = tf.reduce_mean(sumprobs)
    if ispositive:
        loss = - tf.math.log(sumprobs)
    else:
        sumprobs2 = tf.maximum(sumprobs, EPS)
        loss = tf.math.log(sumprobs2)
    loss = tf.reduce_mean(loss)
    return loss, sumprobs, probs

def log_prp_loss(pred, real, ispositive):
    k = tf.cast(tf.reduce_sum(real, axis=-1, keepdims=True), tf.float32)
    probs = real * pred
    # print("probs", probs.numpy())
    logprobs = real * tf.math.log(EPS+pred)
    sumprobs = tf.reduce_sum(probs, axis=-1)
    sumprobs = tf.reduce_mean(tf.reduce_sum(probs, axis=-1))

    if ispositive:
        invprob = 1.0 - tf.reduce_sum(probs, axis=-1, keepdims=True)
        invprob2 = tf.maximum(EPS, invprob)
        log_n = tf.math.log(invprob2)
        log_d = tf.reduce_sum(logprobs, axis=-1, keepdims=True) / k
        loss = log_n - log_d
        loss *= tf.stop_gradient(invprob)
    else:
        invprob = tf.reduce_sum(probs, axis=-1, keepdims=True)
        invprob2 = tf.maximum(EPS, invprob)
        log_n = tf.math.log(invprob2)
        
        logprobs = tf.maximum(-20, logprobs)
        log_d = tf.reduce_sum(logprobs, axis=-1, keepdims=True) / k
        loss  = log_d + log_n

    # print("logn", tf.reduce_mean(log_n))
    # print("logd", tf.reduce_mean(log_d))
    loss *= k
    loss = tf.reduce_mean(loss)
    return loss, sumprobs, probs

@tf.custom_gradient
def log_prp_loss2(pred, real, ispositive):
    k = tf.cast(tf.reduce_sum(real, axis=-1, keepdims=True), tf.float32)
    probs = real * pred
    logprobs = real * tf.math.log(EPS+pred)
    
    invprob = 1.0 - tf.reduce_sum(probs, axis=-1, keepdims=True)
    invprob2 = tf.maximum(EPS, invprob)
    log_n = tf.math.log(invprob2)
    
    log_d = tf.reduce_sum(logprobs, axis=-1, keepdims=True) / k
    loss = log_n - log_d
    loss = k * loss
    loss = tf.reduce_mean(loss)
    if not ispositive:
        loss *= -1

    def grad(upstream):
        # grad = p^2/invprob - 1/k
        # grad *= k
        g = (- probs / invprob2) - 1.0 / k
        g *= real
        g *= k
        if not ispositive:
            g *= -1
        return upstream * g, tf.constant(0.0), tf.constant(0.0)

    return loss, grad

def build_model(num_classes,sizes, optimizer_type, lr, softmax=True):
    model = tf.keras.Sequential()
    model.add(tf.keras.layers.Embedding(100, sizes[0], input_length=1, trainable=False))
    model.add(tf.keras.layers.Flatten())
    for s in sizes[1:]:
        model.add(tf.keras.layers.Dense(s, activation='relu'))
    if softmax:
        model.add(tf.keras.layers.Dense(num_classes, activation='softmax'))
    else:
        model.add(tf.keras.layers.Dense(num_classes, activation='linear'))
    model.compile()
    model.summary()

    if optimizer_type == "adam":
        optimizer = tf.keras.optimizers.Adam(lr)
    elif optimizer_type == "sgd":
        optimizer = tf.keras.optimizers.SGD(lr)

    # optimizer = tf.keras.optimizers.Adamax(LR, beta_1=0.3, beta_2=0.9, epsilon=1e-7)

    return model, optimizer

def loss_function(predictions, out, loss_type, ispositive):
    if loss_type == "nll":
        loss, probs, seq_probs = nll_loss(predictions, out, ispositive)
    elif loss_type == "loglin":
        loss, probs, seq_probs = linear_loss(predictions, out, ispositive, "both")
    elif loss_type == "loglin_up":
        loss, probs, seq_probs = linear_loss(predictions, out, ispositive, "up")
    elif loss_type == "loglin_down":
        loss, probs, seq_probs = linear_loss(predictions, out, ispositive, "down")
    else:
        loss, probs, seq_probs = log_prp_loss(predictions, out, ispositive)
    if loss_type == "prp2":
        loss = log_prp_loss2(predictions, out, ispositive)
    return loss, probs, seq_probs


def update(model, optimizer, inp, out, ninp, nout, loss_type, neg_weight):
    with tf.GradientTape() as tape:
        predictions = model(inp)
        ploss, pprobs, pseq_probs = loss_function(predictions, out, loss_type, True)
        npredictions = model(ninp)
        nloss, nprobs, nseq_probs = loss_function(npredictions, nout, loss_type, False)
        loss = ploss + neg_weight * nloss
        
    gradients = tape.gradient(loss, model.trainable_variables)
    optimizer.apply_gradients(zip(gradients, model.trainable_variables))
    return loss, (ploss, pprobs, pseq_probs), (nloss, nprobs, nseq_probs)


def train(model, optimizer, data, ndata, batch_size, epochs, loss_type, neg_weight, suffix):
    train_loss = tf.keras.metrics.Mean(name='train_loss')
    train_loss_pos = tf.keras.metrics.Mean(name='train_loss_pos')
    train_loss_neg = tf.keras.metrics.Mean(name='train_loss_neg')
    train_probs_pos = tf.keras.metrics.Mean(name='train_probs_pos')
    train_probs_neg = tf.keras.metrics.Mean(name='train_probs_neg')

    monitor = MonitorProbs(enabled=MONITOR)

    for e in range(epochs):
        T0 = time.time()
        print("Epoch: ", e)
        train_loss.reset_states()
        train_loss_pos.reset_states()
        train_loss_neg.reset_states()
        train_probs_pos.reset_states()
        train_probs_neg.reset_states()
        batches = make_batches(data[0], data[1], batch_size)
        nbatches = make_batches(ndata[0], ndata[1], batch_size)

        for (inp, out), (ninp, nout) in zip(batches, nbatches):
            loss, (ploss, pprobs, pseq_probs), (nloss, nprobs, nseq_probs) = update(model, optimizer, inp, out, ninp, nout, loss_type, neg_weight)
            train_loss(loss)
            train_loss_pos(ploss)
            train_loss_neg(nloss)
            train_probs_pos(pprobs)
            train_probs_neg(nprobs)
            monitor.update_mlp(inp, out, pseq_probs)

            # print("loss", loss.numpy())
            # print("probs", probs.numpy())
            # for i, p in zip(inp, seq_probs):
                # p2 = np.floor(p.numpy() * 1000) / 1000
                # print(f'   {i} -> {p2}')
        T = time.time() - T0
        print(f'{T} sec, Loss: {train_loss.result():.4f}   {train_loss_pos.result():.4f}/{train_loss_neg.result():.4f} Probs {train_probs_pos.result():.3f}/{train_probs_neg.result():.3f}')
        sys.stdout.flush()
        if train_probs_pos.result() == 1.0:
            break
        
    monitor.plot("probchange_{}.png".format(suffix), k=1, ratios=True, showsum=True)


def evaluate(model, data, tokenizer):

    pos_dict = data["pos_dict"]
    neg_dict = data["neg_dict"]
    posneg_dict = data["posneg_dict"]

    count = 0
    top10 = 0
    top5 = 0
    top1 = 0
    
    for triple in pos_dict:
        pos_paths = pos_dict[triple]
        negatives = posneg_dict[triple]
        neg_paths_list = [neg_dict[tuple(triple)] for triple in negatives]


        # evaluate model on pred
        # we are assuming that this is the same for all the negs!!!
        inp = tokenizer.input2index(triple)
        inp = np.array([[float(inp)]])
        predictions = model(inp).numpy()[0]
        
        pos_paths = [tokenizer.output2index(p) for p in pos_paths]
        pos_prob = np.sum(predictions[pos_paths])
        neg_probs = []
        for neg_paths in neg_paths_list:
            neg_paths = [tokenizer.output2index(p) for p in neg_paths]
            neg_prob = np.sum(predictions[neg_paths])
            neg_probs.append(neg_prob)
        misclassified = np.sum(neg_probs > pos_prob)
        if misclassified < 10:
            top10 += 1
        if misclassified < 5:
            top5 += 1
        if misclassified < 1:
            top1 += 1
        count +=1

    print("TOP1: {}, TOP5: {}, TOP10: {}".format(top1/count, top5/count, top10/count))


def run(config):
    with open(config["datadir"]+"/train", 'rb') as f:
        traindata = pickle.load(f)
    with open(config["datadir"]+"/dev", 'rb') as f:
        evaldata = pickle.load(f)

    pos_dict = traindata["pos_dict"]
    neg_dict = traindata["neg_dict"]

    tokenizer = Tokenizer()
    d = tokenizer.tokenize_dict(pos_dict)
    nd = tokenizer.tokenize_dict(neg_dict)        
    num_classes = len(tokenizer.vocab_out)    

    print("Positives: ", len(d))
    print("Negatives: ", len(nd))
    print("Num classes: ", num_classes)
        
    model, optimizer = build_model(num_classes,
                                   config["network_sizes"],
                                   config["optimizer_type"],
                                   config["lr"],
                                   softmax=config["softmax"]
    )
    data = prepare_data(d, num_classes)
    print("Positive inputs: {}".format(len(data[0])))
    ndata= prepare_data(nd, num_classes)
    print("Negative inputs: {}".format(len(ndata[0])))
    
    train(model, optimizer, data, ndata,
          config["batch_size"],
          config["EPOCHS"],
          config["LOSS_TYPE"],
          config["NEG_WEIGHT"],
          "{}_{}".format(config["exp"], config["LOSS_TYPE"])
    )

    print("Eval on train set")
    evaluate(model, traindata, tokenizer)

    print("Eval on eval set")
    evaluate(model, evaldata, tokenizer)

config1 = {
    "exp": 1,
    "datadir": "out/countries_S1",
    "LOSS_TYPE": "prp",
    "EPOCHS": 30,
    "batch_size": 100,
    "lr": 0.01,
    "optimizer_type": "sgd",
    "network_sizes": (40,40,40),
    "softmax": True,
    "NEG_WEIGHT": 1.0,
}

run(config1)
