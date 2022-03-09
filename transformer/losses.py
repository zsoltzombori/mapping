import tensorflow as tf

@tf.custom_gradient
def LogSumExp(x, axis, mask):
    y = tf.math.log(1e-5 + tf.reduce_sum(mask * tf.math.exp(x), axis=axis))
    y = tf.clip_by_value(y, -10000, 0)
    # y = tf.math.reduce_logsumexp(x, axis=axis)

    def grad(upstream):
        x2 = x - tf.reduce_max(x, axis=axis, keepdims=True)
        e_x = tf.exp(x2) * mask
        softmax = e_x / (1e-5 + tf.reduce_sum(e_x, axis=axis, keepdims=True))
        softmax *= mask
        return tf.expand_dims(upstream, -1) * softmax, tf.constant(0.0), tf.constant(0.0)

    return y, grad


# probability ratio preserving (prp) loss
# probs is (bs * support_size)
def prp_loss(probs, mask_nonzero):
    # loss = (1 - sum probs) / prod(pow(probs, 1/k))
    k = 1.0 * tf.reduce_sum(mask_nonzero, axis=-1, keepdims=True)
    probs = probs * mask_nonzero
    n = tf.maximum(0.0, 1.0 - tf.reduce_sum(probs, axis=-1))
    d = tf.reduce_prod(tf.math.pow(probs, 1.0 / k), axis=-1)
    loss = n/(1e-5 * d)
    return loss


@tf.custom_gradient
def LogOneMinusSumExp(logp, mask):
    # y = log(1-sum(exp(logp)))
    probs = mask * tf.math.exp(logp)
    invprob = 1.0 - tf.reduce_sum(probs, axis=-1, keepdims=True)
    y = tf.math.log(1e-5 + invprob)
    y = tf.clip_by_value(y, -10000, 0)

    def grad(upstream):
        # grad = 1/(1-sum(exp(logp))) * -exp(logp)
        coeff = 1.0 / (1e-5 + invprob)
        probs2 = tf.expand_dims(probs, axis=-1)
        grad = - coeff * probs
        return upstream * grad, tf.constant(0.0)

    return y, grad


# log probability ratio preserving (prp) loss
# log probs is (bs * support_size)
def log_prp_loss(logprobs, mask_nonzero):
    # loss = (1 - sum probs) / prod(pow(probs, 1/k))
    # log loss = log(1-sum(exp(logprobs))) - sum(logprobs)/k
    k = 1.0 * tf.reduce_sum(mask_nonzero, axis=-1, keepdims=True)

    log_n = LogOneMinusSumExp(logprobs, mask_nonzero)      
    log_d = tf.reduce_sum(logprobs, axis=-1) / k
    loss = log_n - log_d
    loss = k * loss
    return loss


def loss_function(real, pred, ispositive, loss_type):
    mask_zero = tf.math.equal(real, 0)
    mask_nonzero = tf.math.logical_not(mask_zero)
    mask_zero = tf.cast(mask_zero, dtype=pred.dtype)
    mask_nonzero = tf.cast(mask_nonzero, dtype=pred.dtype)
    mask_nonzero_sequence = tf.reduce_max(mask_nonzero, axis=2)

    logprobs = pred - tf.math.reduce_logsumexp(pred, axis=-1, keepdims=True) #(support * bs * seq * tokens)
    
    # focus on the logprobs of real sequence
    logprobs = tf.gather(logprobs, real, batch_dims=3) #(support * bs * seq)
    # print("logprob", tf.transpose(logprobs, perm=[1,0,2])[0])

    # replace padding element probs with 1 for multiplication
    logprobs *= mask_nonzero
    sequence_logprobs = tf.reduce_sum(logprobs, axis=2) #(support * bs)

    sequence_logprobs = tf.transpose(sequence_logprobs, perm=[1,0]) # (bs * support)
    mask_nonzero_sequence = tf.transpose(mask_nonzero_sequence, perm=[1,0]) # (bs * support)

    # reduce logprobs for all supporting sequences, removing padding sequences
    datapoint_logprobs = LogSumExp(sequence_logprobs, -1, mask_nonzero_sequence) #(bs,)
    datapoint_probs = tf.math.exp(datapoint_logprobs) #(bs,)

    if loss_type=="nll":
        if ispositive:
            loss = - datapoint_logprobs
        else:
            loss = tf.maximum(0.0, datapoint_logprobs + 30.0)
            
    elif loss_type=="prp": # probability ratio preserving (prp) loss
        # loss = (1 - sum probs) / prod(pow(probs, 1/k))
        sequence_probs = tf.math.exp(sequence_logprobs)
        loss = prp_loss(sequence_probs, mask_nonzero_sequence)
        if not ispositive:
            loss = - loss
        
    elif loss_type=="lprp": # # log probability ratio preserving (prp) loss
        # loss = (1 - sum probs) / prod(pow(probs, 1/k))
        # log loss = log(1-sum(exp(logprobs))) - sum(logprobs)/k
        loss = log_prp_loss(sequence_logprobs, mask_nonzero_sequence)
        if not ispositive:
            loss = tf.maximum(0.0, -loss + 30.0)
    else:
        assert False, "Unknown loss type" + loss_type
            

    loss = tf.reduce_mean(loss)
    probs = tf.reduce_mean(datapoint_probs)
    return loss, probs
