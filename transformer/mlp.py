import numpy as np
import tensorflow as tf

LR=0.01
EPOCHS=50
BATCH_SIZE=3
LOSS_TYPE="lprp" #nll, lprp, lprp2
PRETRAIN=10

data_pretrain = [
    (1, (2,3)),
    ]

data = [
    (1, (1,2)),
    (1, (1,3)),
    ]

# count num_classes
num_classes = 0
for d in data + data_pretrain:
    out = d[1]
    num_classes = max(num_classes, max(out))
num_classes += 1
print("Output space size: ", num_classes)

def prepare_data(data):    
    inputs = []
    outputs = []
    for d in data:
        inputs.append(float(d[0]))
        o = [int(x in d[1]) for x in range(num_classes)]
        outputs.append(o)
    return inputs, outputs

inputs, outputs = prepare_data(data)
inputs_pretrain, outputs_pretrain = prepare_data(data_pretrain)


    
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



optimizer = tf.keras.optimizers.Adamax(LR, beta_1=0.3, beta_2=0.9, epsilon=1e-7)
optimizer = tf.keras.optimizers.Adam(LR)
train_loss = tf.keras.metrics.Mean(name='train_loss')
train_probs = tf.keras.metrics.Mean(name='train_probs')

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


# Create the model
model = tf.keras.Sequential()
model.add(tf.keras.layers.Embedding(5, 10, input_length=1))
model.add(tf.keras.layers.Flatten())
model.add(tf.keras.layers.Dense(350, input_shape=(1, ), activation='relu'))
model.add(tf.keras.layers.Dense(50, activation='relu'))
model.add(tf.keras.layers.Dense(num_classes, activation='softmax'))
model.compile()
model.summary()

for e in range(EPOCHS):
    print("Epoch: ", e)
    train_loss.reset_states()
    train_probs.reset_states()
    if e < PRETRAIN:
        # take data from data_pretrain
        batches = make_batches(inputs_pretrain, outputs_pretrain, BATCH_SIZE)
    else:
        batches = make_batches(inputs, outputs, BATCH_SIZE)

    for (inp, out) in batches:
        with tf.GradientTape() as tape:
            predictions = model(inp)
            if LOSS_TYPE == "nll":
                loss, probs, seq_probs = nll_loss(predictions, out)
            else:
                loss, probs, seq_probs = log_prp_loss(predictions, out)
            if LOSS_TYPE == "lprp2":
                loss = log_prp_loss2(predictions, out)
            train_loss(loss)
            train_probs(probs)

            # print("loss", loss.numpy())
            # print("probs", probs.numpy())
            for i, p in zip(inp, seq_probs):
                p2 = np.floor(p.numpy() * 1000) / 1000
                print(f'   {i} -> {p2}')
        gradients = tape.gradient(loss, model.trainable_variables)
        optimizer.apply_gradients(zip(gradients, model.trainable_variables))

    print(f'Loss: {train_loss.result():.4f}, Probs {train_probs.result():.3f}')
