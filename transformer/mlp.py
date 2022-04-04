import numpy as np
import tensorflow as tf
import time
import sys

from monitor import MonitorProbs
import mlp_data

LR=0.001
EPOCHS=20
BATCH_SIZE=40
LOSS_TYPE="nll" #nll, prp, prp2
PRETRAIN=1
MONITOR=False
NEG_WEIGHT=3.0
EPS=1e-5

d1 = [
    (1, (7,8,9)),
]
dp1 = [
    (1, (7,)),
]
d2 = [
    (1, (7,8)),
    (1, (7,9)),
]
dp2 = [
    (1, (8,9)),
]

d3 = [
    (1, (0,1,2)),
    (1, (0,1,3)),
]
dp3 = [
    (1, (1,)),
]

nd3 = [
    (1, (1,4)),
]


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
    logprobs = real * tf.math.log(EPS+pred)
    sumprobs = tf.reduce_sum(probs, axis=-1)
    sumprobs = tf.reduce_mean(tf.reduce_sum(probs, axis=-1))

    if ispositive:
        invprob = 1.0 - tf.reduce_sum(probs, axis=-1, keepdims=True)
        invprob = tf.maximum(EPS, invprob)
        log_n = tf.math.log(invprob)
        log_d = tf.reduce_sum(logprobs, axis=-1, keepdims=True) / k
        loss = log_n - log_d
    else:
        invprob = tf.reduce_sum(probs, axis=-1, keepdims=True)
        invprob = tf.maximum(EPS, invprob)
        log_n = tf.math.log(invprob)
        
        logprobs = tf.maximum(-20, logprobs)
        log_d = tf.reduce_sum(logprobs, axis=-1, keepdims=True) / k
        loss  = log_d + log_n

    print("logn", tf.reduce_mean(log_n))
    print("logd", tf.reduce_mean(log_d))
    loss *= k
    loss = tf.reduce_mean(loss)
    return loss, sumprobs, probs

@tf.custom_gradient
def log_prp_loss2(pred, real, ispositive):
    k = tf.cast(tf.reduce_sum(real, axis=-1, keepdims=True), tf.float32)
    probs = real * pred
    logprobs = real * tf.math.log(pred)
    
    invprob = 1.0 - tf.reduce_sum(probs, axis=-1, keepdims=True)
    invprob += EPS
    log_n = tf.math.log(invprob)
    log_d = tf.reduce_sum(logprobs, axis=-1, keepdims=True) / k
    loss = log_n - log_d
    loss = k * loss
    loss = tf.reduce_mean(loss)
    if not ispositive:
        loss *= -1

    def grad(upstream):
        # grad = p^2/invprob - 1/k
        # grad *= k
        g = (- probs / invprob) - 1.0 / k
        g *= real
        g *= k
        if not ispositive:
            g *= -1
        return upstream * g, tf.constant(0.0)

    return loss, grad

def build_model(num_classes):
    model = tf.keras.Sequential()
    model.add(tf.keras.layers.Embedding(100, 10, input_length=1))
    model.add(tf.keras.layers.Flatten())
    model.add(tf.keras.layers.Dense(350, input_shape=(1, ), activation='relu'))
    model.add(tf.keras.layers.Dense(10, activation='relu'))
    model.add(tf.keras.layers.Dense(num_classes, activation='softmax'))
    model.compile()
    model.summary()

    # optimizer = tf.keras.optimizers.Adamax(LR, beta_1=0.3, beta_2=0.9, epsilon=1e-7)
    optimizer = tf.keras.optimizers.Adam(LR)

    return model, optimizer

def loss_function(predictions, out, loss_type, ispositive):
    if loss_type == "nll":
        loss, probs, seq_probs = nll_loss(predictions, out, ispositive)
    else:
        loss, probs, seq_probs = log_prp_loss(predictions, out, ispositive)
    if loss_type == "prp2":
        loss = log_prp_loss2(predictions, out, ispositive)
    return loss, probs, seq_probs

def update(model, optimizer, inp, out, loss_type):
    with tf.GradientTape() as tape:
        predictions = model(inp)
        loss, probs, seq_probs = loss_function(predictions, out, loss_type, True)
    gradients = tape.gradient(loss, model.trainable_variables)
    optimizer.apply_gradients(zip(gradients, model.trainable_variables))
    print("loss: ", loss.numpy())
    print("gradient sum: ", [tf.reduce_sum(g).numpy() for g in gradients])
    return loss, (loss, probs, seq_probs), (tf.zeros_like(loss), tf.zeros_like(probs), tf.zeros_like(seq_probs))

def update_neg(model, optimizer, inp, out, nout, loss_type):
    with tf.GradientTape() as tape:
        predictions = model(inp)
        ploss, pprobs, pseq_probs = loss_function(predictions, out, loss_type, True)
        nloss, nprobs, nseq_probs = loss_function(predictions, nout, loss_type, False)
        loss = ploss + NEG_WEIGHT * nloss
        
    gradients = tape.gradient(loss, model.trainable_variables)
    optimizer.apply_gradients(zip(gradients, model.trainable_variables))
    return loss, (ploss, pprobs, pseq_probs), (nloss, nprobs, nseq_probs)


def train(model, optimizer, data, epochs, loss_type, suffix, ndata=None):
    train_loss = tf.keras.metrics.Mean(name='train_loss')
    train_loss_pos = tf.keras.metrics.Mean(name='train_loss_pos')
    train_loss_neg = tf.keras.metrics.Mean(name='train_loss_neg')
    train_probs_pos = tf.keras.metrics.Mean(name='train_probs_pos')
    train_probs_neg = tf.keras.metrics.Mean(name='train_probs_neg')

    monitor = MonitorProbs(enabled=MONITOR)

    for e in range(epochs):
        T0 = time.time()
        print("Epoch: ", e)
        evaluate(model, data, ndata=ndata)
        train_loss.reset_states()
        train_loss_pos.reset_states()
        train_loss_neg.reset_states()
        train_probs_pos.reset_states()
        train_probs_neg.reset_states()
        batches = make_batches(data[0], data[1], BATCH_SIZE)

        for (inp, out) in batches:
            if ndata is None:
                loss, (ploss, pprobs, pseq_probs), (nloss, nprobs, nseq_probs) = update(model, optimizer, inp, out, loss_type)
                train_loss(loss)
                train_loss_pos(ploss)
                train_probs_pos(pprobs)
                monitor.update_mlp(inp, out, pseq_probs)
            else:
                nbatches = make_batches(ndata[0], ndata[1], BATCH_SIZE)
                for (_, nout) in nbatches:
                    if nout.shape[0] != out.shape[0]:
                        break
                    loss, (ploss, pprobs, pseq_probs), (nloss, nprobs, nseq_probs) = update_neg(model, optimizer, inp, out, nout, loss_type)
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
        
    monitor.plot("probchange_{}.png".format(suffix), k=1, ratios=False)

def evaluate(model, data, ndata=None):

    if ndata is not None:
        nout = ndata[1]
        nout = tf.math.reduce_max(nout, axis=0)

    batches = make_batches(data[0], data[1], 1)
    count = 0
    pos = 0
    neg = 0
    for (inp, out) in batches:
        predictions = model(inp)
        best = tf.math.argmax(predictions, axis=-1)
        for b, o in zip(best, out):
            count += 1
            if b in o:
                pos += 1
            if (ndata is not None) and (nout[b] == 1):
                neg += 1

    print("EVALUATION: Count: {}, Pos: {}, Neg: {}".format(count, pos/count, neg/count))


def run(exp):
    if exp==1:
        d = d1
        dp = dp1
        num_classes=10
        PRETRAIN=1
        LOSS_TYPE="nll"
        
    elif exp==2:
        d = d1
        dp = dp1
        num_classes=10
        PRETRAIN=1
        LOSS_TYPE="prp"

    if exp==3:
        d = d2
        dp = dp2
        num_classes=10
        PRETRAIN=100
        LOSS_TYPE="nll"
        
    elif exp==4:
        d = d2
        dp = dp2
        num_classes=10
        PRETRAIN=100
        LOSS_TYPE="prp"

    elif exp==5:
        d=d3
        dp=dp3
        nd=nd3
        num_classes=10
        PRETRAIN=30
        LOSS_TYPE="prp"

    elif exp==6:
        d=d3
        dp=dp3
        nd=nd3
        num_classes=10
        PRETRAIN=30
        LOSS_TYPE="nll"

    elif exp==7:
        datadir="outdata/cmt_renamed/cmt_renamed"
        data, vocab_in, vocab_out = mlp_data.load_data(datadir)
        d = data["pos"]
        nd = None #data["neg"]
        dp=None
        num_classes = len(vocab_out)
        PRETRAIN=0
        LOSS_TYPE="prp"

    elif exp==8:
        datadir="outdata/cmt_renamed/cmt_renamed"
        data, vocab_in, vocab_out = mlp_data.load_data(datadir)
        d = data["pos"]
        nd = None #data["neg"]
        dp=None
        num_classes = len(vocab_out)
        PRETRAIN=0
        LOSS_TYPE="nll"

    model, optimizer = build_model(num_classes)
    if PRETRAIN>0:
        data_pretrain = prepare_data(dp, num_classes)
        train(model, optimizer, data_pretrain, PRETRAIN, LOSS_TYPE, "{}_{}_pre".format(exp, LOSS_TYPE))
    data = prepare_data(d, num_classes)
    print("Positive inputs: {}".format(len(data[0])))
    if nd is not None:
        ndata= prepare_data(nd, num_classes)
        print("Negative inputs: {}".format(len(ndata[0])))
    else:
        ndata = None
    train(model, optimizer, data, EPOCHS, LOSS_TYPE, "{}_{}".format(exp, LOSS_TYPE), ndata=ndata)

#run(1)
#run(2)
#run(3)
#run(4)
#run(5)
#run(6)
run(7)
#run(8)
