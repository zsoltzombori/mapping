from xmlrpc.server import MultiPathXMLRPCServer
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
def prp_loss(probs, mask_nonzero):
    # loss = (1 - sum probs) / prod(pow(probs, 1/k))

    k = 1.0 * tf.reduce_sum(mask_nonzero, axis=-1, keepdims=True)
    n = tf.maximum(0.0, 1.0 - tf.reduce_sum(probs, axis=-1))

    probs_with_ones = (1 - mask_nonzero) + probs
    d = tf.reduce_prod(probs_with_ones, axis=-1)
    d = tf.math.pow(d, 1.0 / k)
    loss = n/(EPS + d)
    loss = loss / (loss+1) # scale [0, inf] to [0,1]
    return loss

# prp squashed to [0,1] with x/(x+1)
def sprp_loss(probs, mask_nonzero):
    k = 1.0 * tf.reduce_sum(mask_nonzero, axis=-1, keepdims=True)
    n = tf.maximum(0.0, 1.0 - tf.reduce_sum(probs, axis=-1))

    probs_with_ones = (1 - mask_nonzero) + probs
    d = tf.reduce_prod(probs_with_ones, axis=-1)
    d = tf.math.pow(d, 1.0 / k)
    d += n
    d = tf.maximum(EPS, d)
    loss = n /d
    return loss
    


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
        log_d = tf.reduce_sum(mask_nonzero * logprobs, axis=-1) / k
        loss = log_n - log_d
        # loss *= 1.0 - sumprob
    else:
        log_n = LogSumExp(logprobs, -1, mask_nonzero)
        logprobs2 = tf.maximum(logEPS, logprobs)
        log_d = tf.reduce_sum(mask_nonzero * logprobs2, axis=-1, keepdims=True) / k
        loss = log_d + log_n
        loss = tf.maximum(loss, -10000)
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


def loss_function(real, pred, ispositive, loss_type, multiplier=None, compute_explicit_targets=False, explicit_targets=None, explicit_target_mask=None):
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
        # loss *= tf.stop_gradient(1.0-datapoint_probs)
        if not ispositive:
            loss = 1.0 - loss
        
    elif loss_type=="lprp": # # log probability ratio preserving (prp) loss
        # loss = (1 - sum probs) / prod(pow(probs, 1/k))
        # log loss = log(1-sum(exp(logprobs))) - sum(logprobs)/k
        probs = tf.nn.softmax(pred, axis=-1)
        print("Probs (batch_idx=0):")
        print(tf.squeeze(probs[:,0,:,:]))

        loss = log_prp_loss(sequence_logprobs, mask_nonzero_sequence, ispositive)

    elif loss_type=="slprp": # sigmoid of lprp loss
        loss = log_prp_loss(sequence_logprobs, mask_nonzero_sequence, True)
        loss = tf.math.sigmoid(loss/100)
        if not ispositive:
            loss = 1.0 - loss
    elif loss_type=="sprp": # squashed prp loss (equivalent to slprp, with different stability behaviour)
        loss = sprp_loss(sequence_probs, mask_nonzero_sequence)
        if not ispositive:
            loss = 1.0 - loss
    elif loss_type=="seq_prp": # sequencial prp updates
        from sequential.utils import seq_prp_targets
        ALPHA = 1.05
        TOKENS = 6
        EMPTY=0

        # real shape (support * bs * seq)
        # pred shape (support * bs * seq * tokens)
        
        loss = 0
        batch_size = tf.get_static_value(tf.shape(real)[1])
        probs = tf.nn.softmax(pred, axis=-1)

        target_list = []
        target_shape = pred.get_shape()
        targets = tf.zeros(target_shape)
        if compute_explicit_targets:
            for b in range(batch_size): # TODO: Parallelize the loop!!!
                # Get target updates: 
                sequences_b = tf.get_static_value(tf.squeeze(real[:,b,:]))
                probabilities_b = tf.get_static_value(tf.squeeze(probs[:,b,:,:]))
                targets_b = seq_prp_targets(sequences_b, probabilities_b, TOKENS, ALPHA)
                target_list.append(targets_b)

            targets = tf.transpose(tf.stack(target_list), perm=[1,0,2,3])
            targets_mask = tf.math.not_equal(probs, targets)
        else:
            assert explicit_targets is not None, "Explicit targets need to be provided, if not computed."
            assert explicit_target_mask is not None, "Explicit target mask needs to be provided, if targets are not computed."
            targets = explicit_targets
            targets_mask = explicit_target_mask

        # prob_weights = []
        # for b in range(batch_size): 
        #     sequences_b = tf.get_static_value(tf.squeeze(real[:,b,:]))
        #     prob_weights.append(get_prob_weights(sequences_b))
        seq_elements = tf.cast(tf.reshape(tf.get_static_value(real), [-1]), tf.int64)  #flatten
        prob_weights = tf.cast(tf.math.not_equal(seq_elements, tf.cast(tf.fill(seq_elements.shape, EMPTY), tf.int64)), tf.float32)
        # loss = tf.nn.l2_loss(weights*(probs[targets_mask] - targets[targets_mask]))
        # categorical cross entropy
        loss = (-1)*prob_weights*targets[targets_mask]*tf.nn.log_softmax(probs[targets_mask])
    else:
        assert False, "Unknown loss type" + loss_type

    probs = tf.reduce_mean(datapoint_probs)
    loss = tf.reduce_mean(loss)

    if loss_type=="seq_prp":
        # Return explicit targets to fit + mask
        return loss, probs, sequence_probs, targets, targets_mask
    else:
        return loss, probs, sequence_probs    
        

def loss_function_joint(real_pos, real_neg, pred_pos, pred_neg):
    pos_logprobs, pos_mask = get_sequence_logprobs(real_pos, pred_pos)
    neg_logprobs, neg_mask = get_sequence_logprobs(real_neg, pred_neg)

    pos_log_d = tf.reduce_sum(pos_logprobs * pos_mask, axis=-1)
    neg_log_d = tf.reduce_sum(neg_logprobs * neg_mask, axis=-1)

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
