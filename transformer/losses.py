import tensorflow as tf

@tf.custom_gradient
def LogSumExp(x, axis, mask):
    y = tf.math.log(tf.reduce_sum(mask * tf.math.exp(x), axis=axis))
    y = tf.clip_by_value(y, -10000, 0)
    # y = tf.math.reduce_logsumexp(x, axis=axis)

    def grad(upstream):
        x2 = x - tf.reduce_max(x, axis=axis)
        e_x = tf.exp(x2) * mask
        softmax = e_x / (1e-10 + tf.reduce_sum(e_x, axis=axis))
        softmax *= mask
        return upstream * softmax, tf.constant(0.0), tf.constant(0.0)

    return y, grad

def loss_function(real, pred, ispositive):
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
    # print("sequence_logprobs", tf.transpose(sequence_logprobs, perm=[1,0])[0])

    if False: # old loss function
        # reduce logprobs for all supporting sequences, removing padding sequences
        sequence_logprobs_all = LogSumExp(sequence_logprobs, 0, mask_nonzero_sequence)
        # print("sequence_logprobs_all", sequence_logprobs_all)

        sequence_probs_all = tf.math.exp(sequence_logprobs_all)
        # print("sequence_probs_all", sequence_probs_all)
    
        if ispositive:
            loss = - sequence_logprobs_all
        else:
            loss = tf.maximum(0.0, sequence_logprobs_all + 30.0)

    else: # new loss function
        sequence_probs = tf.math.exp(sequence_logprobs) * mask_nonzero_sequence
        sequence_probs_all = tf.reduce_sum(sequence_probs, axis=0)
        if ispositive:
            # print(sequence_probs_all)
            # print("logprobs", sequence_logprobs)
            # print("probs", sequence_probs)
            seq_weight = tf.stop_gradient(1.0 - sequence_probs_all)
            loss = - tf.reduce_sum(sequence_logprobs, axis=0)
        else:
            seq_weight = tf.stop_gradient(sequence_probs_all)
            loss = tf.maximum(0.0, tf.reduce_sum(sequence_logprobs, axis=0) + 30.0)
        loss *= seq_weight

    loss = tf.reduce_mean(loss)
    probs = tf.reduce_mean(sequence_probs_all)
    return loss, probs





# def loss_function(real, pred, ispositive):
#   pos_loss_ = loss_object(real, pred)

#   mask = tf.math.logical_not(tf.math.equal(real, 0))
#   mask = tf.cast(mask, dtype=pos_loss_.dtype)

#   pos_loss_ *= mask
#   pos_loss = tf.reduce_sum(pos_loss_, axis=-1)/tf.reduce_sum(mask, axis=-1)
#   pos_loss = ispositive * pos_loss

#   neg_loss_ = loss_for_negatives(real, pred)
#   neg_loss_ *= mask
#   # neg_loss = tf.reduce_sum(neg_loss_, axis=-1)/tf.reduce_sum(mask, axis=-1)
#   neg_loss = tf.reduce_min(neg_loss_, axis=-1)
#   neg_loss = (1-ispositive) * neg_loss

  
#   # ent_loss_ = entropy_loss(pred)
#   # ent_loss_ *= mask
#   # ent_loss = tf.reduce_sum(ent_loss_, axis=-1)/tf.reduce_sum(mask, axis=-1)
#   # neg_loss = tf.maximum(-NEG_CLIP, (ispositive - 1) * loss)
  
#   loss = pos_loss + NEG_WEIGHT * neg_loss #+ ENT_WEIGHT * ent_loss
#   loss = tf.reduce_mean(loss)
#   pos_loss = tf.reduce_mean(pos_loss)
#   neg_loss = tf.reduce_mean(neg_loss)
#   return pos_loss, neg_loss, loss



# loss_object2 = tf.keras.losses.BinaryCrossentropy(
#     from_logits=True, reduction='none')

# def funny_loss_function(real, pred, ispositive):
#   mask = tf.math.logical_not(tf.math.equal(real, 0))

#   # binary crossentropy on the target logit
#   target_logits = tf.map_fn(my_gather_list, (pred, real), fn_output_signature=tf.TensorSpec(shape=[None], dtype=tf.float32))
#   target_logits = tf.expand_dims(target_logits, -1)
#   loss_ = loss_object2(tf.ones_like(target_logits), target_logits)

#   loss_ += loss_object(real, pred)

#   # small force pulling all logits to the opposite direction
#   reg_loss_ = tf.reduce_mean(pred, axis=-1)
#   # loss_ = loss_ + 0.001 * reg_loss_
  
#   mask = tf.cast(mask, dtype=loss_.dtype)
#   loss_ *= mask

#   loss = tf.reduce_sum(loss_, axis=-1)/tf.reduce_sum(mask, axis=-1)
#   pos_loss = ispositive * loss
#   neg_loss = tf.maximum(-NEG_CLIP, (ispositive - 1) * loss)
#   loss = pos_loss + NEG_WEIGHT * neg_loss
#   loss = tf.reduce_mean(loss)
#   pos_loss = tf.reduce_mean(pos_loss)
#   neg_loss = tf.reduce_mean(neg_loss)
#   return pos_loss, neg_loss, loss


# def loss_for_negatives(real, pred):
#   loss_ = loss_object(real, pred)
#   loss_ = tf.maximum(NEG_CLIP, loss_)
#   return - loss_

#   # real_onehot = tf.one_hot(real, pred.shape[-1])
#   # pred_masked_real = pred - real_onehot * 1e10

#   # sampled_target = tf.map_fn(my_sampler, pred_masked_real, fn_output_signature=tf.TensorSpec(shape=[None], dtype=tf.int64))
#   # loss_ = loss_object(sampled_target, pred)

#   # for i in range(9):
#   #   sampled_target = tf.map_fn(my_sampler, pred_masked_real, fn_output_signature=tf.TensorSpec(shape=[None], dtype=tf.int64))
#   #   loss_ += loss_object(sampled_target, pred)

#   # loss_ /= 10.0
#   # # loss_ -= loss_object(real, pred)

#   # # probs = tf.nn.softmax(pred)
#   # # loss_ = real_onehot * tf.abs((1-real_onehot)-probs)
#   # # loss_ = tf.reduce_sum(loss_, axis=2)
#   # return loss_

# # loss_type = 1: only positives
# # loss_type = -1: only negatives
# def loss_function_selective(real, pred, ispositive, loss_type):
#   loss_ = loss_object(real, pred)

#   mask = tf.math.logical_not(tf.math.equal(real, 0))
#   mask = tf.cast(mask, dtype=loss_.dtype)

#   loss_ *= mask
#   # loss = tf.reduce_sum(loss_, axis=-1)/tf.reduce_sum(mask, axis=-1)
#   loss = tf.reduce_sum(loss_, axis=-1)

#   pos_loss = loss * ispositive * (loss_type + 1.0) / 2
#   neg_loss = loss * (ispositive-1.0) * (loss_type - 1.0) / 2
#   loss = pos_loss + neg_loss

#   return pos_loss, neg_loss, loss


# loss_object = tf.keras.losses.SparseCategoricalCrossentropy(
#     from_logits=True, reduction='none')

# def entropy_loss(pred):
#   # pred is (batch_size * seq_len * out_vocab_size)
#   probs = tf.nn.softmax(pred)
#   entropy = - tf.reduce_sum(probs * tf.math.log(probs), axis=-1)
#   return entropy

# def my_sampler(logits):
#   sample = tf.random.categorical(logits, 1)
#   sample = sample[:,0]
#   return sample

