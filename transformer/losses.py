import tensorflow as tf

EPS=1e-20
logEPS=tf.math.log(EPS)

@tf.custom_gradient
def LogSumExp(x, axis, mask):
    x_max = tf.reduce_max(x, axis=axis, keepdims=True)
    x2 = x - x_max
    e_x = tf.math.exp(x2) * mask
    sum_e_x = tf.reduce_sum(e_x, axis=axis, keepdims=True)
    sum_e_x += EPS
    y = x_max + tf.math.log(sum_e_x)

    def grad(upstream):
        softmax = e_x / sum_e_x
        softmax *= mask
        return upstream * softmax, tf.constant(0.0), tf.constant(0.0)

    return y, grad


# # probability ratio preserving (prp) loss
# # probs is (bs * support_size)
# def prp_loss(probs, mask_nonzero):
#     # loss = (1 - sum probs) / prod(pow(probs, 1/k))
#     k = 1.0 * tf.reduce_sum(mask_nonzero, axis=-1, keepdims=True)
#     probs = probs * mask_nonzero
#     n = tf.maximum(0.0, 1.0 - tf.reduce_sum(probs, axis=-1))
#     d = tf.reduce_prod(tf.math.pow(probs, 1.0 / k), axis=-1)
#     loss = n/(1e-5 * d)
#     return loss


@tf.custom_gradient
def LogOneMinusSumExp(logp, mask):
    # y = log(1-sum(exp(logp)))
    probs = mask * tf.math.exp(logp)
    invprob = 1.0 - tf.reduce_sum(probs, axis=-1, keepdims=True)
    invprob2 = tf.maximum(EPS, invprob)
    log_n = tf.math.log(invprob2)


    def grad(upstream):
        # grad = 1/(1-sum(exp(logp))) * -exp(logp)
        coeff = 1.0 / invprob2
        coeff *= tf.where(invprob > EPS, 1.0, 0.0)
        
        g = - coeff * probs
        return upstream * g, tf.constant(0.0)

    return log_n, grad


# log probability ratio preserving (prp) loss
# log probs is (bs * support_size)
def log_prp_loss(logprobs, mask_nonzero, ispositive):
    # loss = (1 - sum probs) / prod(pow(probs, 1/k))
    # log loss = log(1-sum(exp(logprobs))) - sum(logprobs)/k
    k = 1.0 * tf.reduce_sum(mask_nonzero, axis=-1, keepdims=True)
    # sumprob = tf.stop_gradient(tf.reduce_sum(tf.math.exp(logprobs), axis=-1, keepdims=False))
            
    if ispositive:
        log_n = LogOneMinusSumExp(logprobs, mask_nonzero)
        log_d = tf.reduce_sum(logprobs, axis=-1) / k
        loss = log_n - log_d
        # loss *= 1.0 - sumprob
    else:
        log_n = LogSumExp(logprobs, -1, mask_nonzero)
        logprobs2 = tf.maximum(logEPS, logprobs)
        log_d = tf.reduce_sum(logprobs2, axis=-1, keepdims=True) / k
        loss = log_d + log_n
        # loss *= sumprob

    loss = k * loss
    return loss

def get_sequence_logprobs(real, pred):
    mask_zero = tf.math.equal(real, 0)
    mask_nonzero = tf.math.logical_not(mask_zero)
    mask_zero = tf.cast(mask_zero, dtype=pred.dtype)
    mask_nonzero = tf.cast(mask_nonzero, dtype=pred.dtype)
    mask_nonzero_sequence = tf.reduce_max(mask_nonzero, axis=2)

    logprobs = pred - tf.math.reduce_logsumexp(pred, axis=-1, keepdims=True) #(support * bs * seq * tokens)
    
    # focus on the logprobs of real sequence
    logprobs = tf.gather(logprobs, real, batch_dims=3) #(support * bs * seq)

    # replace padding element probs with 1 for multiplication
    logprobs *= mask_nonzero
    sequence_logprobs = tf.reduce_sum(logprobs, axis=2) #(support * bs)

    sequence_logprobs = tf.transpose(sequence_logprobs, perm=[1,0]) # (bs * support)
    mask_nonzero_sequence = tf.transpose(mask_nonzero_sequence, perm=[1,0]) # (bs * support)

    return sequence_logprobs, mask_nonzero_sequence


def loss_function(real, pred, ispositive, loss_type):
    sequence_logprobs, mask_nonzero_sequence = get_sequence_logprobs(real, pred)
    
    sequence_probs = tf.math.exp(sequence_logprobs) * mask_nonzero_sequence

    # reduce logprobs for all supporting sequences, removing padding sequences
    datapoint_logprobs = LogSumExp(sequence_logprobs, -1, mask_nonzero_sequence) #(bs * support)
    datapoint_logprobs = tf.reduce_sum(datapoint_logprobs, axis=-1) # (bs, )
    datapoint_probs = tf.reduce_sum(sequence_probs, axis=-1)

    if loss_type=="nll":
        if ispositive:
            loss = - datapoint_logprobs
        else:
            datapoint_logprobs2 = tf.maximum(logEPS, datapoint_logprobs)
            loss = datapoint_logprobs2
            
    elif loss_type=="prp": # probability ratio preserving (prp) loss
        # loss = (1 - sum probs) / prod(pow(probs, 1/k))
        loss = prp_loss(sequence_probs, mask_nonzero_sequence)
        loss *= tf.stop_gradient(1.0-datapoint_probs)
        if not ispositive:
            loss = - loss
        
    elif loss_type=="lprp": # # log probability ratio preserving (prp) loss
        # loss = (1 - sum probs) / prod(pow(probs, 1/k))
        # log loss = log(1-sum(exp(logprobs))) - sum(logprobs)/k
        loss = log_prp_loss(sequence_logprobs, mask_nonzero_sequence, ispositive)
    else:
        assert False, "Unknown loss type" + loss_type
            

    loss = tf.reduce_mean(loss)
    probs = tf.reduce_mean(datapoint_probs)
    return loss, probs, sequence_probs    
        

def loss_function_joint(real_pos, real_neg, pred_pos, pred_neg):
    print(real_pos)
    print(real_neg)
    xxx
    pos_logprobs, pos_mask = get_sequence_logprobs(real_pos, pred_pos)
    neg_logprobs, neg_mask = get_sequence_logprobs(real_neg, pred_neg)

    pos_log_d = tf.reduce_sum(pos_logprobs, axis=-1)
    neg_log_d = tf.reduce_sum(neg_logprobs, axis=-1)

    pos_k = 1.0 * tf.reduce_sum(pos_mask, axis=-1, keepdims=True)
    neg_k = 1.0 * tf.reduce_sum(neg_mask, axis=-1, keepdims=True)

    logprobs = tf.concat([pos_logprobs, neg_logprobs], axis=-1)
    mask = tf.concat([pos_mask, neg_mask], axis=-1)

    log_n = LogOneMinusSumExp(logprobs, mask)

    loss = (pos_k - neg_k) * log_n - pos_log_d + neg_log_d
    loss = tf.reduce_mean(loss)

    pos_sequence_probs = tf.math.exp(pos_logprobs) * pos_mask
    pos_datapoint_probs = tf.reduce_sum(pos_sequence_probs, axis=-1)
    pos_probs = tf.reduce_mean(pos_datapoint_probs)
    neg_sequence_probs = tf.math.exp(neg_logprobs) * neg_mask
    neg_datapoint_probs = tf.reduce_sum(neg_sequence_probs, axis=-1)
    neg_probs = tf.reduce_mean(neg_datapoint_probs)

    return loss, pos_probs, neg_probs
