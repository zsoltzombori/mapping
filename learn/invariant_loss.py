import numpy as np
import tensorflow as tf

def invariant_loss(values):
    n = tf.constant(-1.0)
    d = tf.constant(1.0)
    for v in values:
        n += v
        d *= tf.sqrt(v)
    loss = - n/d
    return loss

def exp1(steps, lr=0.1):
    p1 = 0.3
    p2 = 0.1
    p3 = 1 - p1 - p2

    probs = tf.constant([p1, p2, p3])
    logits = tf.math.log(probs)

    for i in range(steps):
        with tf.GradientTape() as tape:
            tape.watch(logits)
            probs = tf.nn.softmax(logits)
            loss = invariant_loss([probs[0], probs[1]])
        print(loss)
        gradients = tape.gradient(loss, logits)
        print("-------", i, "--------")
        print("logits:", logits)
        print("probs: ", probs)
        print("gradients: ", gradients)
        logits -= gradients * lr

def exp2(steps, lr=0.1):
    p1 = 0.3
    p2 = 0.1
    p3 = 0.2
    p4 = 1 - p1 - p2 - p3

    probs = tf.constant([p1, p2, p3, p4])
    logits = tf.math.log(probs)

    for i in range(steps):
        with tf.GradientTape() as tape:
            tape.watch(logits)
            probs = tf.nn.softmax(logits)
            loss = invariant_loss([probs[0], probs[1], probs[2]])
            print(loss)
        gradients = tape.gradient(loss, logits)
        print("-------", i, "--------")
        print("logits:", logits)
        print("probs: ", probs)
        print("gradients: ", gradients)
    logits -= gradients * lr


def exp3(steps, lr=0.1):
    pa = 0.6
    pb = 0.2
    pc = 1.0 - pa - pb
    pab = 0.5
    paa = 0.5
    pba = 0.5
    pbb = 0.5

    prob_e = tf.constant([pa, pb, pc])
    prob_a = tf.constant([paa, pab])
    prob_b = tf.constant([pba, pbb])
    logits_e = tf.math.log(prob_e)
    logits_a = tf.math.log(prob_a)
    logits_b = tf.math.log(prob_b)

    for i in range(steps):
        with tf.GradientTape() as tape:
            tape.watch(logits_e)
            tape.watch(logits_a)
            tape.watch(logits_b)
            
            probs_e = tf.nn.softmax(logits_e)
            probs_a = tf.nn.softmax(logits_a)
            probs_b = tf.nn.softmax(logits_b)

            probs_aa = probs_e[0] * probs_a[0]
            probs_ab = probs_e[0] * probs_a[1]
            probs_ba = probs_e[1] * probs_b[0]
            probs_bb = probs_e[1] * probs_b[1]
            
            loss = invariant_loss([probs_ab, probs_ba])            
            print("loss: ", loss)
        gradients = tape.gradient(loss, [logits_e, logits_a, logits_b])
        print("-------", i, "--------")
        print("probs_e: ", probs_e)
        print("probs_a: ", probs_a)
        print("probs_b: ", probs_b)
        print("probs_ab, probs_ba: ", probs_ab.numpy(), probs_ba.numpy())
        print("gradients: ", gradients)
        logits_e -= gradients[0] * lr
        logits_a -= gradients[1] * lr
        logits_b -= gradients[2] * lr

print("Experiment 1")
exp1(10)

print("Experiment 2")
exp2(10)

print("Experiment 3")
exp3(10)





