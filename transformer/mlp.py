import numpy as np
import tensorflow as tf

from monitor import MonitorProbs

LR=0.003
EPOCHS=300
BATCH_SIZE=10
LOSS_TYPE="nll" #nll, prp, prp2
PRETRAIN=1
MONITOR=True

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
    

def prepare_data(data, data_pretrain):
    num_classes = 0
    for d in data + data_pretrain:
        out = d[1]
        num_classes = max(num_classes, max(out))
    num_classes += 1
    # print("Output space size: ", num_classes)

    def aux(data):    
        inputs = []
        outputs = []
        for d in data:
            inputs.append(float(d[0]))
            o = [int(x in d[1]) for x in range(num_classes)]
            outputs.append(o)
        return inputs, outputs

    inputs, outputs = aux(data)
    inputs_pretrain, outputs_pretrain = aux(data_pretrain)
    return(inputs, outputs), (inputs_pretrain, outputs_pretrain), num_classes


    
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


def nll_loss(pred, real):
    probs = real * pred
    sumprobs = tf.reduce_sum(probs, axis=-1)
    loss = - tf.math.log(sumprobs)
    loss = tf.reduce_mean(loss)
    sumprobs = tf.reduce_mean(sumprobs)
    return loss, sumprobs, probs

def log_prp_loss(pred, real):
    k = tf.cast(tf.reduce_sum(real, axis=-1, keepdims=True), tf.float32)
    probs = real * pred
    logprobs = real * tf.math.log(pred)

    invprob = 1.0 - tf.reduce_sum(probs, axis=-1, keepdims=True)
    invprob += 1e-20
    log_n = tf.math.log(invprob)
    log_d = tf.reduce_sum(logprobs, axis=-1, keepdims=True) / k
    loss = log_n - log_d
    loss *= k
    
    sumprobs = tf.reduce_sum(probs, axis=-1)
    loss = tf.reduce_mean(loss)
    sumprobs = tf.reduce_mean(tf.reduce_sum(probs, axis=-1))
    return loss, sumprobs, probs

@tf.custom_gradient
def log_prp_loss2(pred, real):
    k = tf.cast(tf.reduce_sum(real, axis=-1, keepdims=True), tf.float32)
    probs = real * pred
    logprobs = real * tf.math.log(pred)
    
    invprob = 1.0 - tf.reduce_sum(probs, axis=-1, keepdims=True)
    invprob += 1e-20
    log_n = tf.math.log(invprob)
    log_d = tf.reduce_sum(logprobs, axis=-1, keepdims=True) / k
    loss = log_n - log_d
    loss = k * loss
    loss = tf.reduce_mean(loss)

    def grad(upstream):
        # grad = p^2/invprob - 1/k
        # grad *= k
        g = (- probs / invprob) - 1.0 / k
        g *= real
        g *= k
        return upstream * g, tf.constant(0.0)

    return loss, grad

def build_model(num_classes):
    model = tf.keras.Sequential()
    model.add(tf.keras.layers.Embedding(5, 10, input_length=1))
    model.add(tf.keras.layers.Flatten())
    # model.add(tf.keras.layers.Dense(350, input_shape=(1, ), activation='relu'))
    model.add(tf.keras.layers.Dense(50, activation='relu'))
    model.add(tf.keras.layers.Dense(num_classes, activation='softmax'))
    model.compile()
    model.summary()
    return model

def train(model, data, data_pretrain, pretrain_epochs, loss_type, suffix):
    optimizer = tf.keras.optimizers.Adamax(LR, beta_1=0.3, beta_2=0.9, epsilon=1e-7)
    optimizer = tf.keras.optimizers.Adam(LR)
    train_loss = tf.keras.metrics.Mean(name='train_loss')
    train_probs = tf.keras.metrics.Mean(name='train_probs')

    if MONITOR:
        monitor = MonitorProbs()

    for e in range(EPOCHS):
        print("Epoch: ", e)
        train_loss.reset_states()
        train_probs.reset_states()
        if e < pretrain_epochs:
            # take data from data_pretrain
            batches = make_batches(data_pretrain[0], data_pretrain[1], BATCH_SIZE)
            pretrain_curr = True
            print("pretrain")
        else:
            batches = make_batches(data[0], data[1], BATCH_SIZE)
            pretrain_curr = False

        for (inp, out) in batches:
            with tf.GradientTape() as tape:
                predictions = model(inp)
                if loss_type == "nll":
                    loss, probs, seq_probs = nll_loss(predictions, out)
                else:
                    loss, probs, seq_probs = log_prp_loss(predictions, out)
                if loss_type == "prp2":
                    loss = log_prp_loss2(predictions, out)
                train_loss(loss)
                train_probs(probs)

                if MONITOR and not pretrain_curr:
                    print("update", e)
                    monitor.update_mlp(inp, out, seq_probs)

                # print("loss", loss.numpy())
                # print("probs", probs.numpy())
                for i, p in zip(inp, seq_probs):
                    p2 = np.floor(p.numpy() * 1000) / 1000
                    print(f'   {i} -> {p2}')
            gradients = tape.gradient(loss, model.trainable_variables)
            optimizer.apply_gradients(zip(gradients, model.trainable_variables))

        print(f'Loss: {train_loss.result():.4f}, Probs {train_probs.result():.3f}')
    if MONITOR:
        monitor.plot("probchange_{}.png".format(suffix), k=1, ratios=False)

def run(exp):
    if exp==1:
        d = d1
        dp = dp1
        PRETRAIN=1
        LOSS_TYPE="nll"
        
    elif exp==2:
        d = d1
        dp = dp1
        PRETRAIN=1
        LOSS_TYPE="prp"

    if exp==3:
        d = d2
        dp = dp2
        PRETRAIN=100
        LOSS_TYPE="nll"
        
    elif exp==4:
        d = d2
        dp = dp2
        PRETRAIN=100
        LOSS_TYPE="prp"

    data, data_pretrain, num_classes = prepare_data(d, dp)
    model = build_model(num_classes)
    train(model, data, data_pretrain, PRETRAIN, LOSS_TYPE, "{}_{}".format(exp, LOSS_TYPE))

#run(1)
#run(2)
run(3)
run(4)
