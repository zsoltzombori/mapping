import numpy as np
import tensorflow as tf
import time
import sys

from monitor import MonitorProbs
import mlp_data
import losses

LR=0.1
LOSS_TYPE="nll" #nll, prp, prp2
MONITOR=True

EPS=1e-5
logEPS=tf.math.log(EPS)


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

def linear_loss(pred, real, ispositive, which="both"):
    # we are assuming that pred is the logit vector not the prob vector!!!
    logits_up = real * pred
    logits_down = (1-real) * pred

    if which == "both":
        loss = tf.reduce_sum(logits_down - logits_up, axis=-1)
    elif which == "up":
        loss = tf.reduce_sum(- logits_up, axis=-1)
    elif which == "down":
        loss = tf.reduce_sum(logits_down, axis=-1)

    if not ispositive:
        loss *= -1
    return loss

def nll_loss(pred, real, ispositive):
    probs = real * pred
    sumprobs = tf.reduce_sum(probs, axis=-1)
    if ispositive:
        loss = - tf.math.log(sumprobs)
    else:
        sumprobs2 = tf.maximum(sumprobs, EPS)
        loss = tf.math.log(sumprobs2)
    loss = tf.reduce_mean(loss)
    return loss



def log_prp_loss(pred, real, ispositive):
    k = tf.cast(tf.reduce_sum(real, axis=-1, keepdims=True), tf.float32)
    probs = real * pred
    # print("probs", probs.numpy())
    logprobs = real * tf.math.log(EPS+pred)

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
    return loss

def biprp_loss(logits, probs, real, ispositive, logit_decay=0.01):
    k = tf.cast(tf.reduce_sum(real, axis=-1, keepdims=True), tf.float32)
    k_neg = tf.cast(tf.reduce_sum(1-real, axis=-1, keepdims=True), tf.float32)

    pos_weight = -1/k
    neg_weight = 1/k_neg

    logprobs = tf.math.log(EPS+probs)
    
    pos_loss = tf.reduce_sum(pos_weight * logprobs * real, axis=-1)
    neg_loss = tf.reduce_sum(neg_weight * logprobs * (1-real), axis=-1)
    loss = neg_loss + pos_loss
    
    if not ispositive:
        loss = - loss
    loss += logit_decay * tf.reduce_sum(tf.square(logits), axis=-1)
    loss = tf.reduce_mean(loss)
    return loss

def lws_loss(logits, real, ispositive, beta=1.0, logit_decay=0.0):
    explogits = tf.math.exp(logits)
    explogits_neg = tf.math.exp(-logits)

    pos_weight = real * explogits / tf.reduce_sum(real*explogits)
    neg_weight = (1-real) * explogits / tf.reduce_sum((1-real) * explogits)

    pos_probs = 1/(1+explogits) # sigmoid
    neg_probs = 1/(1+explogits_neg) # sigmoid

    # print("pos_probs", pos_probs)
    # print("neg_probs", neg_probs)


    pos_loss = tf.reduce_sum(pos_weight * pos_probs * real, axis=-1)
    neg_loss = tf.reduce_sum(neg_weight * neg_probs * (1-real), axis=-1)

    loss = pos_loss + beta * neg_loss

    # print("=----------")
    # print("logits", logits)
    # print("explogits", explogits)
    # print(loss)
    
    if not ispositive:
        loss = - loss
    loss += logit_decay * tf.reduce_sum(tf.square(logits), axis=-1)
    loss = tf.reduce_mean(loss)
    return loss


def rc_loss(pred, real, ispositive):
    probs = real * pred
    logprobs = real * tf.math.log(EPS+pred)
    normalised_probs = probs / tf.reduce_sum(probs, axis=-1, keepdims=True)
    weights = 1/2 * tf.stop_gradient(normalised_probs)
    weighted_logprobs = weights * logprobs
    if ispositive:
        loss = - tf.reduce_sum(weighted_logprobs, axis=-1)
    else:
        weighted_logprobs = tf.maximum(weighted_logprobs, logEPS)
        loss = tf.reduce_sum(weighted_logprobs, axis=-1)
    return loss

# meritocratic loss
@tf.custom_gradient
def meritocratic_loss(pred, real, beta):
    probs = real * pred
    sumprobs = tf.reduce_sum(probs, axis=-1)
    loss = - tf.math.log(sumprobs)

    def grad(upstream):
        q_mml = probs / tf.reduce_sum(probs, axis=-1, keepdims=True)
        q_beta = real * q_mml ** beta
        q_beta = q_beta / (EPS + tf.reduce_sum(q_beta, axis=-1, keepdims=True))
        logprobs = real * tf.math.log(EPS+pred)
        g = q_beta * logprobs
        return tf.expand_dims(upstream,axis=1) * g, tf.constant(0.0), tf.constant(0.0)
        # return upstream * g, tf.constant(0.0), tf.constant(0.0)
                                                        
    return loss, grad


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

def loss_function(predictions, out, loss_type, softmax, ispositive):
    if softmax:
        probs = predictions
    else:
        probs = tf.nn.softmax(predictions, axis=-1)
    probs_allowed = out * probs
    sumprobs_allowed = tf.reduce_mean(tf.reduce_sum(probs_allowed, axis=-1))

    if loss_type == "nll":
        loss = nll_loss(predictions, out, ispositive)
    elif loss_type == "loglin":
        loss = linear_loss(predictions, out, ispositive, "both")
    elif loss_type == "loglin_up":
        loss = linear_loss(predictions, out, ispositive, "up")
    elif loss_type == "loglin_down":
        loss = linear_loss(predictions, out, ispositive, "down")
    elif loss_type == "prp":
        loss = log_prp_loss(predictions, out, ispositive)
    elif loss_type == "prp2":
        loss = log_prp_loss2(predictions, out, ispositive)
    elif loss_type == "rc":
        loss = rc_loss(predictions, out, ispositive)
    elif loss_type == "meritocracy0.0":
        assert ispositive
        loss = meritocratic_loss(predictions, out, 0.0)
    elif loss_type == "meritocracy0.25":
        assert ispositive
        loss = meritocratic_loss(predictions, out, 0.25)
    elif loss_type == "meritocracy0.5":
        assert ispositive
        loss = meritocratic_loss(predictions, out, 0.5)
    elif loss_type == "meritocracy0.75":
        assert ispositive
        loss = meritocratic_loss(predictions, out, 0.75)
    elif loss_type == "meritocracy1.0":
        assert ispositive
        loss = meritocratic_loss(predictions, out, 1.0)
    elif loss_type == "biprp":
        assert softmax == False
        loss = biprp_loss(predictions, probs, out, ispositive)
    elif loss_type == "lws":
        assert softmax == False
        loss = lws_loss(predictions, out, ispositive)
    return loss, sumprobs_allowed, probs_allowed

def update(model, optimizer, inp, out, loss_type, softmax=True):
    with tf.GradientTape() as tape:
        predictions = model(inp)
        loss, probs, seq_probs = loss_function(predictions, out, loss_type, softmax=softmax, ispositive=True)
    gradients = tape.gradient(loss, model.trainable_variables)
    optimizer.apply_gradients(zip(gradients, model.trainable_variables))
    # print("loss: ", loss.numpy())
    # print("gradient sum: ", [tf.reduce_sum(g).numpy() for g in gradients])
    return loss, (loss, probs, seq_probs), (tf.zeros_like(loss), tf.zeros_like(probs), tf.zeros_like(seq_probs))

def update_neg(model, optimizer, inp, out, nout, loss_type, neg_weight, softmax=True):
    with tf.GradientTape() as tape:
        predictions = model(inp)
        ploss, pprobs, pseq_probs = loss_function(predictions, out, loss_type, softmax=softmax, ispositive=True)
        nloss, nprobs, nseq_probs = loss_function(predictions, nout, loss_type, softmax=softmax, ispositive=False)
        loss = ploss + neg_weight * nloss
        
    gradients = tape.gradient(loss, model.trainable_variables)
    optimizer.apply_gradients(zip(gradients, model.trainable_variables))
    return loss, (ploss, pprobs, pseq_probs), (nloss, nprobs, nseq_probs)


def train(model, optimizer, data, batch_size, epochs, loss_type, neg_weight, suffix, ratios, ndata=None, keymap={}, softmax=True):
    train_loss = tf.keras.metrics.Mean(name='train_loss')
    train_loss_pos = tf.keras.metrics.Mean(name='train_loss_pos')
    train_loss_neg = tf.keras.metrics.Mean(name='train_loss_neg')
    train_probs_pos = tf.keras.metrics.Mean(name='train_probs_pos')
    train_probs_neg = tf.keras.metrics.Mean(name='train_probs_neg')

    monitor = MonitorProbs(enabled=MONITOR)

    for e in range(epochs):
        T0 = time.time()
        # print("Epoch: ", e)
        # evaluate(model, data, ndata=ndata)
        train_loss.reset_states()
        train_loss_pos.reset_states()
        train_loss_neg.reset_states()
        train_probs_pos.reset_states()
        train_probs_neg.reset_states()
        batches = make_batches(data[0], data[1], batch_size)

        for (inp, out) in batches:
            if ndata is None:
                loss, (ploss, pprobs, pseq_probs), (nloss, nprobs, nseq_probs) = update(model, optimizer, inp, out, loss_type, softmax=softmax)
                train_loss(loss)
                train_loss_pos(ploss)
                train_probs_pos(pprobs)
                monitor.update_mlp(inp, out, pseq_probs)
            else:
                nbatches = make_batches(ndata[0], ndata[1], batch_size)
                for (_, nout) in nbatches:
                    if nout.shape[0] != out.shape[0]:
                        break
                    loss, (ploss, pprobs, pseq_probs), (nloss, nprobs, nseq_probs) = update_neg(model, optimizer, inp, out, nout, loss_type, neg_weight, softmax=softmax)
                    train_loss(loss)
                    train_loss_pos(ploss)
                    train_loss_neg(nloss)
                    train_probs_pos(pprobs)
                    train_probs_neg(nprobs)
                    monitor.update_mlp(inp, out, pseq_probs)

            # print("loss", loss.numpy())
            # print("probs", pprobs.numpy())
            # for i, p in zip(inp, seq_probs):
                # p2 = np.floor(p.numpy() * 1000) / 1000
                # print(f'   {i} -> {p2}')
        T = time.time() - T0
        # print(f'{T} sec, Loss: {train_loss.result():.4f}   {train_loss_pos.result():.4f}/{train_loss_neg.result():.4f} Probs {train_probs_pos.result():.3f}/{train_probs_neg.result():.3f}')
        sys.stdout.flush()
        if train_probs_pos.result() == 1.0:
            break
        
    monitor.plot("probchange_{}.pdf".format(suffix), k=1, ratios=ratios, showsum=True, keymap=keymap)
    pos_ratio, neg_ratio = evaluate(model, data, ndata=ndata)
    return pos_ratio, neg_ratio

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

    pos_ratio = pos/count
    neg_ratio = neg/count
    print("EVALUATION: Count: {}, Pos: {}, Neg: {}".format(count, pos_ratio, neg_ratio))
    return (pos_ratio, neg_ratio)


def run(exp):

    ###############
    # default values
    keymap = {}
    repeat = 1
    ###############
    
    if exp==1: # figure 1a
        d = [(0, (0,1,2))]
        dp = [(0, (0,)),(0, (0,)), (0,(1,)), (0, (0,1))]
        nd=None
        num_classes=100
        LOSS_TYPE="nll"
        EPOCHS = 200
        PRETRAIN=5
        batch_size=1
        lr=0.1
        optimizer_type="sgd"
        # network_sizes=(10,10)
        network_sizes=(10,)
        softmax=True
        NEG_WEIGHT=3
        ratios=True
        keymap={0:'A', 1:'B', 2:'C'}
        repeat=1
        
        
    elif exp==2: # figure 1b
        d = [(0, (0,1,2))]
        dp = [(0, (0,)),(0, (0,)), (0,(1,)), (0, (0,1))]
        nd=None
        num_classes=100
        LOSS_TYPE="prp"
        EPOCHS = 200
        PRETRAIN=5
        batch_size=1
        lr=0.1
        optimizer_type="sgd"
        # network_sizes=(10,10)
        network_sizes=(10,)
        softmax=True
        NEG_WEIGHT=3
        ratios=True
        keymap={0:'A', 1:'B', 2:'C'}
        repeat=1

    elif exp==3: # figure 1b with zero layers
        d = [(0, (0,1,2))]
        dp = [(0, (0,))]
        nd=None
        num_classes=10
        LOSS_TYPE="prp"
        EPOCHS = 200
        PRETRAIN=3
        batch_size=1
        lr=0.1
        optimizer_type="sgd"
        network_sizes=(10,)
        softmax=True
        NEG_WEIGHT=3
        ratios=True


    if exp==4: # figure 3 left
        d = [(0, (0,1)), (0, (0,2))]
        dp = [(0, (1,2))]
        nd=None
        num_classes=100
        LOSS_TYPE="nll"
        EPOCHS = 200
        PRETRAIN=20
        batch_size=2
        lr=0.1
        optimizer_type="sgd"
        network_sizes=(10,10)
        softmax=True
        NEG_WEIGHT=3
        ratios=False
        keymap={0:'A', 1:'B', 2:'C'}
        repeat=10


    elif exp==5: # figure 3 right
        d = [(0, (0,1)), (0, (0,2))]
        dp = [(0, (1,2))]
        nd=None
        num_classes=100
        LOSS_TYPE="prp"
        EPOCHS=200
        PRETRAIN=20
        batch_size=2
        lr=0.1
        optimizer_type="sgd"
        network_sizes=(10,10)
        softmax=True
        NEG_WEIGHT=3
        ratios=False
        keymap={0:'A', 1:'B', 2:'C'}
        repeat=10

    elif exp==6: # figure 4 consistent 1
        d = [(0, (0,1,2,3,4,5,6,7,8,9)),
             (0, (0,1,2,3,4,5,6,7,8,10)),
             (0, (0,1,2,3,4,5,6,7,9,10)),
             (0, (0,1,2,3,4,5,6,8,9,10)),
             (0, (0,1,2,3,4,5,7,8,9,10)),
             (0, (0,1,2,3,4,6,7,8,9,10)),
             (0, (0,1,2,3,5,6,7,8,9,10)),
             (0, (0,1,2,4,5,6,7,8,9,10)),
             (0, (0,1,3,4,5,6,7,8,9,10)),
             (0, (0,2,3,4,5,6,7,8,9,10)),
        ]
        dp = [(0, (1,2,3,4,5,6,7,8,9,10))]
        nd=None
        num_classes=100
        LOSS_TYPE="prp"
        EPOCHS=100
        PRETRAIN=30
        batch_size=5
        lr=0.1
        optimizer_type="sgd"
        network_sizes=(10,10)
        softmax=True
        NEG_WEIGHT=3
        ratios=False
        keymap={}
        for i in range(11):
            keymap[i]=r'$o_{{{}}}$'.format(i)
        repeat=10

    elif exp==7: # figure 4 consistent 2
        d = [(0, (0,1,2,3,4,5,6,7,8,9)),
             (0, (0,1,2,3,4,5,6,7,8,10)),
             (0, (0,1,2,3,4,5,6,7,9,10)),
             (0, (0,1,2,3,4,5,6,8,9,10)),
             (0, (0,1,2,3,4,5,7,8,9,10)),
             (0, (0,1,2,3,4,6,7,8,9,10)),
             (0, (0,1,2,3,5,6,7,8,9,10)),
             (0, (0,1,2,4,5,6,7,8,9,10)),
             (0, (0,1,3,4,5,6,7,8,9,10)),
             (0, (0,2,3,4,5,6,7,8,9,10)),
        ]
        dp = [(0, (1,2,3,4,5,6,7,8,9,10))]
        nd=None
        num_classes=100
        LOSS_TYPE="nll"
        EPOCHS=100
        PRETRAIN=30
        batch_size=5
        lr=0.1
        optimizer_type="sgd"
        network_sizes=(10,10)
        softmax=True
        NEG_WEIGHT=3
        ratios=False
        keymap={}
        for i in range(11):
            keymap[i]=r'$o_{{{}}}$'.format(i)
        repeat=10
        
    elif exp==8: # figure 4 inconsistent prp
        d = [(0, (1,2,3)),
             (0, (1,2,4)),
             (0, (1,2,5)),
             (0, (1,3,5)),
             (0, (3,5,6)),
        ]
        dp = None
        nd=None
        num_classes=100
        LOSS_TYPE="prp"
        EPOCHS=200
        PRETRAIN=0
        batch_size=5
        lr=0.1
        optimizer_type="sgd"
        network_sizes=(10,10)
        softmax=True
        NEG_WEIGHT=3
        ratios=False

    elif exp==9: # figure 4 inconsistent nll
        d = [(0, (1,2,3)),
             (0, (1,2,4)),
             (0, (1,2,5)),
             (0, (1,3,5)),
             (0, (3,5,6)),
        ]
        dp = None
        nd=None
        num_classes=100
        LOSS_TYPE="nll"
        EPOCHS=200
        PRETRAIN=0
        batch_size=5
        lr=0.1
        optimizer_type="sgd"
        network_sizes=(10,10)
        softmax=True
        NEG_WEIGHT=3
        ratios=False

    elif exp==10: # consistent prp with 3 outputs
        d = [(0, (0,1)),
             (0, (0,2)),
        ]
        dp = [(0, (1,2))]
        nd=None
        num_classes=3
        LOSS_TYPE="prp"
        EPOCHS=100
        PRETRAIN=30
        batch_size=2
        lr=0.1
        optimizer_type="sgd"
        network_sizes=(10,10)
        softmax=True
        NEG_WEIGHT=3
        ratios=False

    elif exp==11: # consistent nll with 3 outputs
        d = [(0, (0,1)),
             (0, (0,2)),
        ]
        dp = [(0, (2,))]
        nd=None
        num_classes=3
        LOSS_TYPE="nll"
        EPOCHS=200
        PRETRAIN=50
        batch_size=2
        lr=0.1
        optimizer_type="sgd"
        network_sizes=(10,10)
        softmax=True
        NEG_WEIGHT=3
        ratios=False
        
    elif exp==12: # consistent prp with 3 outputs
        d = [(0, (0,1)),
             (0, (0,2)),
        ]
        dp = [(0, (1,2))]
        nd=None
        num_classes=10
        LOSS_TYPE="prp"
        EPOCHS=100
        PRETRAIN=50
        batch_size=2
        lr=0.1
        optimizer_type="sgd"
        network_sizes=(10,10)
        softmax=True
        NEG_WEIGHT=3
        ratios=False

    elif exp==13: # consistent nll with 3 outputs
        d = [(0, (0,1)),
             (0, (0,2)),
        ]
        dp = [(0, (0,2))]
        nd=None
        num_classes=10
        LOSS_TYPE="nll"
        EPOCHS=200
        PRETRAIN=50
        batch_size=2
        lr=0.1
        optimizer_type="sgd"
        network_sizes=(10,10)
        softmax=True
        NEG_WEIGHT=3
        ratios=False
        
    if exp==14: # figure 3 left
        # num classes: 100 -> 10
        d = [(0, (0,1)), (0, (0,2))]
        dp = [(0, (1,2))]
        nd=None
        num_classes=10
        LOSS_TYPE="nll"
        EPOCHS = 200
        PRETRAIN=20
        batch_size=2
        lr=0.1
        optimizer_type="sgd"
        network_sizes=(10,10)
        softmax=True
        NEG_WEIGHT=3
        ratios=False
        keymap={0:'A', 1:'B', 2:'C'}
        repeat=20


    elif exp==15: # figure 3 right
        # num classes: 100 -> 10
        d = [(0, (0,1)), (0, (0,2))]
        dp = [(0, (1,2))]
        nd=None
        num_classes=10
        LOSS_TYPE="prp"
        EPOCHS=200
        PRETRAIN=20
        batch_size=2
        lr=0.1
        optimizer_type="sgd"
        network_sizes=(10,10)
        softmax=True
        NEG_WEIGHT=3
        ratios=False
        keymap={0:'A', 1:'B', 2:'C'}
        repeat=10

    elif exp==16: # figure 4 consistent 2
        d = [(0, (0,1,2,3,4,5,6,7,8,9)),
             (0, (0,1,2,3,4,5,6,7,8,10)),
             (0, (0,1,2,3,4,5,6,7,9,10)),
             (0, (0,1,2,3,4,5,6,8,9,10)),
             (0, (0,1,2,3,4,5,7,8,9,10)),
             (0, (0,1,2,3,4,6,7,8,9,10)),
             (0, (0,1,2,3,5,6,7,8,9,10)),
             (0, (0,1,2,4,5,6,7,8,9,10)),
             (0, (0,1,3,4,5,6,7,8,9,10)),
             (0, (0,2,3,4,5,6,7,8,9,10)),
        ]
        dp = [(0, (1,2,3,4,5,6,7,8,9,10))]
        nd=None
        num_classes=100
        LOSS_TYPE="rc"
        EPOCHS=100
        PRETRAIN=30
        batch_size=5
        lr=0.1
        optimizer_type="sgd"
        network_sizes=(10,10)
        softmax=True
        NEG_WEIGHT=3
        ratios=False
        keymap={}
        for i in range(11):
            keymap[i]="o_{}".format(i)
        repeat=1000

    elif exp==17: # figure 4 consistent 2
        d = [(0, (0,1,2,3,4,5,6,7,8,9)),
             (0, (0,1,2,3,4,5,6,7,8,10)),
             (0, (0,1,2,3,4,5,6,7,9,10)),
             (0, (0,1,2,3,4,5,6,8,9,10)),
             (0, (0,1,2,3,4,5,7,8,9,10)),
             (0, (0,1,2,3,4,6,7,8,9,10)),
             (0, (0,1,2,3,5,6,7,8,9,10)),
             (0, (0,1,2,4,5,6,7,8,9,10)),
             (0, (0,1,3,4,5,6,7,8,9,10)),
             (0, (0,2,3,4,5,6,7,8,9,10)),
        ]
        dp = [(0, (1,2,3,4,5,6,7,8,9,10))]
        nd=None
        num_classes=100
        LOSS_TYPE="meritocracy0.0"
        EPOCHS=200
        PRETRAIN=30
        batch_size=5
        lr=0.1
        optimizer_type="sgd"
        network_sizes=(10,10)
        softmax=True
        NEG_WEIGHT=3
        ratios=False
        keymap={}
        for i in range(11):
            keymap[i]=r'$o_{{{}}}$'.format(i)
        repeat=1000

    elif exp==18: # figure 4 consistent 2
        d = [(0, (0,1,2,3,4,5,6,7,8,9)),
             (0, (0,1,2,3,4,5,6,7,8,10)),
             (0, (0,1,2,3,4,5,6,7,9,10)),
             (0, (0,1,2,3,4,5,6,8,9,10)),
             (0, (0,1,2,3,4,5,7,8,9,10)),
             (0, (0,1,2,3,4,6,7,8,9,10)),
             (0, (0,1,2,3,5,6,7,8,9,10)),
             (0, (0,1,2,4,5,6,7,8,9,10)),
             (0, (0,1,3,4,5,6,7,8,9,10)),
             (0, (0,2,3,4,5,6,7,8,9,10)),
        ]
        dp = [(0, (1,2,3,4,5,6,7,8,9,10))]
        nd=None
        num_classes=100
        LOSS_TYPE="meritocracy0.25"
        EPOCHS=200
        PRETRAIN=30
        batch_size=5
        lr=0.1
        optimizer_type="sgd"
        network_sizes=(10,10)
        softmax=True
        NEG_WEIGHT=3
        ratios=False
        keymap={}
        for i in range(11):
            keymap[i]=r'$o_{{{}}}$'.format(i)
        repeat=1000
        
    elif exp==19: # figure 4 consistent 2
        d = [(0, (0,1,2,3,4,5,6,7,8,9)),
             (0, (0,1,2,3,4,5,6,7,8,10)),
             (0, (0,1,2,3,4,5,6,7,9,10)),
             (0, (0,1,2,3,4,5,6,8,9,10)),
             (0, (0,1,2,3,4,5,7,8,9,10)),
             (0, (0,1,2,3,4,6,7,8,9,10)),
             (0, (0,1,2,3,5,6,7,8,9,10)),
             (0, (0,1,2,4,5,6,7,8,9,10)),
             (0, (0,1,3,4,5,6,7,8,9,10)),
             (0, (0,2,3,4,5,6,7,8,9,10)),
        ]
        dp = [(0, (1,2,3,4,5,6,7,8,9,10))]
        nd=None
        num_classes=100
        LOSS_TYPE="meritocracy0.5"
        EPOCHS=200
        PRETRAIN=30
        batch_size=5
        lr=0.1
        optimizer_type="sgd"
        network_sizes=(10,10)
        softmax=True
        NEG_WEIGHT=3
        ratios=False
        keymap={}
        for i in range(11):
            keymap[i]=r'$o_{{{}}}$'.format(i)
        repeat=1000
        
    elif exp==20: # figure 4 consistent 2
        d = [(0, (0,1,2,3,4,5,6,7,8,9)),
             (0, (0,1,2,3,4,5,6,7,8,10)),
             (0, (0,1,2,3,4,5,6,7,9,10)),
             (0, (0,1,2,3,4,5,6,8,9,10)),
             (0, (0,1,2,3,4,5,7,8,9,10)),
             (0, (0,1,2,3,4,6,7,8,9,10)),
             (0, (0,1,2,3,5,6,7,8,9,10)),
             (0, (0,1,2,4,5,6,7,8,9,10)),
             (0, (0,1,3,4,5,6,7,8,9,10)),
             (0, (0,2,3,4,5,6,7,8,9,10)),
        ]
        dp = [(0, (1,2,3,4,5,6,7,8,9,10))]
        nd=None
        num_classes=100
        LOSS_TYPE="meritocracy0.75"
        EPOCHS=200
        PRETRAIN=30
        batch_size=5
        lr=0.1
        optimizer_type="sgd"
        network_sizes=(10,10)
        softmax=True
        NEG_WEIGHT=3
        ratios=False
        keymap={}
        for i in range(11):
            keymap[i]=r'$o_{{{}}}$'.format(i)
        repeat=1000
        
    elif exp==21: # figure 4 consistent 2
        d = [(0, (0,1,2,3,4,5,6,7,8,9)),
             (0, (0,1,2,3,4,5,6,7,8,10)),
             (0, (0,1,2,3,4,5,6,7,9,10)),
             (0, (0,1,2,3,4,5,6,8,9,10)),
             (0, (0,1,2,3,4,5,7,8,9,10)),
             (0, (0,1,2,3,4,6,7,8,9,10)),
             (0, (0,1,2,3,5,6,7,8,9,10)),
             (0, (0,1,2,4,5,6,7,8,9,10)),
             (0, (0,1,3,4,5,6,7,8,9,10)),
             (0, (0,2,3,4,5,6,7,8,9,10)),
        ]
        dp = [(0, (1,2,3,4,5,6,7,8,9,10))]
        nd=None
        num_classes=100
        LOSS_TYPE="meritocracy1.0"
        EPOCHS=200
        PRETRAIN=30
        batch_size=5
        lr=0.1
        optimizer_type="sgd"
        network_sizes=(10,10)
        softmax=True
        NEG_WEIGHT=3
        ratios=False
        keymap={}
        for i in range(11):
            keymap[i]=r'$o_{{{}}}$'.format(i)
        repeat=1000
        
    elif exp==22: # figure 4 consistent 2
        d = [(0, (0,1,2,3,4,5,6,7,8,9)),
             (0, (0,1,2,3,4,5,6,7,8,10)),
             (0, (0,1,2,3,4,5,6,7,9,10)),
             (0, (0,1,2,3,4,5,6,8,9,10)),
             (0, (0,1,2,3,4,5,7,8,9,10)),
             (0, (0,1,2,3,4,6,7,8,9,10)),
             (0, (0,1,2,3,5,6,7,8,9,10)),
             (0, (0,1,2,4,5,6,7,8,9,10)),
             (0, (0,1,3,4,5,6,7,8,9,10)),
             (0, (0,2,3,4,5,6,7,8,9,10)),
        ]
        dp = [(0, (1,2,3,4,5,6,7,8,9,10))]
        nd=None
        num_classes=100
        LOSS_TYPE="biprp"
        EPOCHS=200
        PRETRAIN=30
        batch_size=5
        lr=0.1
        optimizer_type="sgd"
        network_sizes=(10,10)
        softmax=False
        NEG_WEIGHT=3
        ratios=False
        keymap={}
        for i in range(11):
            keymap[i]=r'$o_{{{}}}$'.format(i)
        repeat=1000
        
    elif exp==23: # figure 4 consistent 2
        d = [(0, (0,1,2,3,4,5,6,7,8,9)),
             (0, (0,1,2,3,4,5,6,7,8,10)),
             (0, (0,1,2,3,4,5,6,7,9,10)),
             (0, (0,1,2,3,4,5,6,8,9,10)),
             (0, (0,1,2,3,4,5,7,8,9,10)),
             (0, (0,1,2,3,4,6,7,8,9,10)),
             (0, (0,1,2,3,5,6,7,8,9,10)),
             (0, (0,1,2,4,5,6,7,8,9,10)),
             (0, (0,1,3,4,5,6,7,8,9,10)),
             (0, (0,2,3,4,5,6,7,8,9,10)),
        ]
        dp = [(0, (1,2,3,4,5,6,7,8,9,10))]
        nd=None
        num_classes=100
        LOSS_TYPE="lws"
        EPOCHS=200
        PRETRAIN=30
        batch_size=5
        lr=1.0
        optimizer_type="sgd"
        network_sizes=(10,10)
        softmax=False
        NEG_WEIGHT=3
        ratios=False
        keymap={}
        for i in range(11):
            keymap[i]=r'$o_{{{}}}$'.format(i)
        repeat=1000
        
    # elif exp==5:
    #     d=d3
    #     dp=dp3
    #     nd=nd3
    #     num_classes=10
    #     PRETRAIN=30
    #     LOSS_TYPE="prp"

    # elif exp==6:
    #     d=d3
    #     dp=dp3
    #     nd=nd3
    #     num_classes=10
    #     PRETRAIN=30
    #     LOSS_TYPE="nll"

    # elif exp==7:
    #     datadir="outdata/cmt_renamed/cmt_renamed"
    #     data, vocab_in, vocab_out = mlp_data.load_data(datadir)
    #     d = data["pos"]
    #     nd = None #data["neg"]
    #     dp=None
    #     num_classes = len(vocab_out)
    #     PRETRAIN=0
    #     LOSS_TYPE="prp"

    # elif exp==8:
    #     datadir="outdata/cmt_renamed/cmt_renamed"
    #     data, vocab_in, vocab_out = mlp_data.load_data(datadir)
    #     d = data["pos"]
    #     nd = None #data["neg"]
    #     dp=None
    #     num_classes = len(vocab_out)
    #     PRETRAIN=0
    #     LOSS_TYPE="nll"

    # elif exp==9:
    #     d = [(0, (0,1,2))]
    #     dp = [(0, (0,))]
    #     nd=None
    #     num_classes=10
    #     LOSS_TYPE="loglin"
    #     EPOCHS = 300
    #     PRETRAIN=20
    #     batch_size=1
    #     lr=0.003
    #     optimizer_type="sgd"
    #     network_sizes=(10,10,10,10)
    #     softmax=False
    #     NEG_WEIGHT=3
    #     ratios=True

    # elif exp==10:
    #     d = [(0, (0,1,2))]
    #     dp = [(0, (0,))]
    #     nd=None
    #     num_classes=10
    #     LOSS_TYPE="loglin_up"
    #     EPOCHS = 200
    #     PRETRAIN=20
    #     batch_size=1
    #     lr=0.02
    #     optimizer_type="sgd"
    #     network_sizes=(10,10)
    #     softmax=False
    #     NEG_WEIGHT=3
    #     ratios=True
        
    # elif exp==11:
    #     d = [(0, (0,1,2))]
    #     dp = [(0, (0,))]
    #     nd=None
    #     num_classes=10
    #     LOSS_TYPE="loglin_down"
    #     EPOCHS = 200
    #     PRETRAIN=20
    #     batch_size=1
    #     lr=0.02
    #     optimizer_type="sgd"
    #     network_sizes=(10,10)
    #     softmax=False
    #     NEG_WEIGHT=3
    #     ratios=True

    # elif exp==12:
    #     datadir="outkbc/train"
    #     data, vocab_in, vocab_out = mlp_data.load_data(datadir)
    #     d = data["pos"]
    #     dp=None
    #     nd = data["neg"]
    #     num_classes = len(vocab_out)
    #     LOSS_TYPE="prp"
    #     EPOCHS = 100
    #     PRETRAIN=0
    #     batch_size = 3
    #     lr = 0.1
    #     optimizer_type="sgd"
    #     network_sizes=(10,10)
    #     softmax=True
    #     NEG_WEIGHT=0.1
    #     ratios=True

    pos_ratio_total = 0
    neg_ratio_total = 0
    for i in range(repeat):
        tf.random.set_seed(i)
        np.random.seed(i)
        model, optimizer = build_model(num_classes, network_sizes, optimizer_type, lr, softmax=softmax)
        if PRETRAIN>0:
            PRETRAIN_LOSS_TYPE="nll"
            data_pretrain = prepare_data(dp, num_classes)
            train(model, optimizer, data_pretrain, batch_size, PRETRAIN, PRETRAIN_LOSS_TYPE, NEG_WEIGHT, "pre_{}_{}_{}".format(exp, LOSS_TYPE, i), ratios=ratios, keymap=keymap, softmax=softmax)
        data = prepare_data(d, num_classes)
        print("Positive inputs: {}".format(len(data[0])))
        if nd is not None:
            ndata= prepare_data(nd, num_classes)
            print("Negative inputs: {}".format(len(ndata[0])))
        else:
            ndata = None
        pos_ratio, neg_ratio = train(model, optimizer, data, batch_size, EPOCHS, LOSS_TYPE, NEG_WEIGHT, "{}_{}_{}".format(exp, LOSS_TYPE,i), ndata=ndata, ratios=ratios, keymap=keymap, softmax=softmax)
        pos_ratio_total += pos_ratio
        neg_ratio_total += neg_ratio
    pos_ratio_total /= repeat
    neg_ratio_total /= repeat
    print("EXPERIMENT {}, Pos ratio: {}, Neg ratio: {}".format(exp, pos_ratio_total, neg_ratio_total))


# run(1) # nll toy example in the intro
# run(2) # prp toy example in the intro
run(4) # nll two samples: (A,B), (A,C)
run(5) # prp two samples: (A,B), (A,C)
# run(6) # prp small consistent
# run(7) # nll small consistent
# run(16) # rc small consistent
# run(17) # merit0.0 small consistent
# run(18) # merit0.25 small consistent
# run(19) # merit0.5 small consistent
# run(20) # merit0.75 small consistent
# run(21) # merit1.0 small consistent
# run(22) # biprp small consistent
# run(23) # lws small consistent

