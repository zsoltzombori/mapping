import numpy as np
import tensorflow as tf

import losses


length=10
samples = [
    [0,1,2,3],
    [0,1,2,4,5],
    [0,1,5],
    [0, 3, 6]
]
steps = 1000
lr = tf.constant(0.1)
LOG=True

probs = tf.random.uniform(shape = (length, ))
probs = probs / tf.reduce_sum(probs)

logits = tf.math.log(probs)


print("probs:", probs)
print("logits: ", logits)

for i in range(steps):
    print("-------", i, "--------")
    for s in samples:
        with tf.GradientTape() as tape:
            tape.watch(logits)
            probs = tf.nn.softmax(logits)
            probs_sel = tf.gather(probs, s)
            probs_sel = tf.expand_dims(probs_sel, 0)
            mask_nonzero = tf.ones_like(probs_sel)
            if LOG:
                logprobs_sel = tf.math.log(probs_sel)
                loss = losses.log_prp_loss(logprobs_sel, mask_nonzero)
            else:
                loss = losses.prp_loss(probs_sel, mask_nonzero)
        gradients = tape.gradient(loss, logits)
        # print("sample: ", s)
        print("loss: ", loss)
        print("logits:", logits)
        print("probs: ", probs[0])
        print("gradients: ", gradients)
        
        logits -= gradients * lr

