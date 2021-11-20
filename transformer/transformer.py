# source: https://www.tensorflow.org/text/tutorials/transformer

NEG_CLIP=0.5
NEG_WEIGHT=0.1
ENT_WEIGHT=0.0
CLIP_NORM = 0.1


import collections
import logging
import os
import pathlib
import re
import string
import sys
import time
import copy
import itertools
import numpy as np

os.environ['TF_XLA_FLAGS'] = '--tf_xla_enable_xla_devices'
import tensorflow as tf
from tensorflow.keras.layers.experimental.preprocessing import TextVectorization

print("GPU available: ", tf.config.list_physical_devices('GPU'))
physical_devices = tf.config.list_physical_devices('GPU')
try:
  tf.config.experimental.set_memory_growth(physical_devices[0], True)
except:
  # Invalid device or cannot modify virtual devices once initialized.
  pass
logging.getLogger('tensorflow').setLevel(logging.ERROR)  # suppress warnings


##########################################

def load_data(datadir, buffer_size, split=(0.7, 0.15, 0.15)):
  element_spec = tf.TensorSpec(shape=(3,), dtype=tf.string, name=None)
  examples = tf.data.experimental.load(datadir, element_spec=element_spec)
  df_size = tf.data.experimental.cardinality(examples).numpy()
  print("Dataset size: ", df_size)

  train_size = int(split[0] * df_size)
  val_size = int(split[1] * df_size)
  test_size = int(split[2] * df_size)
  
  examples = examples.shuffle(buffer_size)
  train_examples = examples.take(train_size)
  train_examples = examples.take(df_size) # TODO remove this line
  test_examples = examples.skip(train_size)
  val_examples = test_examples.skip(val_size)
  test_examples = test_examples.take(test_size)

  return (train_examples, val_examples, test_examples)

def create_tokenizer(vocab_size, max_seq_len, train_text):  
  int_vectorize_layer = TextVectorization(
    max_tokens=vocab_size,
    output_mode='int',
    standardize=None,
    output_sequence_length=max_seq_len)
  int_vectorize_layer.adapt(train_text)
  return int_vectorize_layer
  
def make_batches(ds, tokenizer_in, tokenizer_out, buffer_size, batch_size):
  def prepare_data(x):
    text_in = tf.expand_dims(x[:,0], -1)
    text_out = tf.expand_dims(x[:,1], -1)
    text_in = tokenizer_in(text_in)
    text_out = tokenizer_out(text_out)
    ispositive = tf.cast(tf.math.equal(x[:,2], "True"), tf.float32)
    return text_in, text_out, ispositive
  
  return (
    ds
    .cache()
    .shuffle(buffer_size)
    .batch(batch_size)
    .map(prepare_data, num_parallel_calls=tf.data.AUTOTUNE)
    .prefetch(tf.data.AUTOTUNE))


def get_angles(pos, i, d_model):
  angle_rates = 1 / np.power(10000, (2 * (i//2)) / np.float32(d_model))
  return pos * angle_rates

def positional_encoding(position, d_model):
  angle_rads = get_angles(np.arange(position)[:, np.newaxis],
                          np.arange(d_model)[np.newaxis, :],
                          d_model)

  # apply sin to even indices in the array; 2i
  angle_rads[:, 0::2] = np.sin(angle_rads[:, 0::2])

  # apply cos to odd indices in the array; 2i+1
  angle_rads[:, 1::2] = np.cos(angle_rads[:, 1::2])

  pos_encoding = angle_rads[np.newaxis, ...]

  return tf.cast(pos_encoding, dtype=tf.float32)

def create_padding_mask(seq):
  seq = tf.cast(tf.math.equal(seq, 0), tf.float32)

  # add extra dimensions to add the padding
  # to the attention logits.
  return seq[:, tf.newaxis, tf.newaxis, :]  # (batch_size, 1, 1, seq_len)

def create_look_ahead_mask(size):
  mask = 1 - tf.linalg.band_part(tf.ones((size, size)), -1, 0)
  return mask  # (seq_len, seq_len)

def scaled_dot_product_attention(q, k, v, mask):
  """Calculate the attention weights.
  q, k, v must have matching leading dimensions.
  k, v must have matching penultimate dimension, i.e.: seq_len_k = seq_len_v.
  The mask has different shapes depending on its type(padding or look ahead)
  but it must be broadcastable for addition.

  Args:
    q: query shape == (..., seq_len_q, depth)
    k: key shape == (..., seq_len_k, depth)
    v: value shape == (..., seq_len_v, depth_v)
    mask: Float tensor with shape broadcastable
          to (..., seq_len_q, seq_len_k). Defaults to None.

  Returns:
    output, attention_weights
  """

  matmul_qk = tf.matmul(q, k, transpose_b=True)  # (..., seq_len_q, seq_len_k)

  # scale matmul_qk
  dk = tf.cast(tf.shape(k)[-1], tf.float32)
  scaled_attention_logits = matmul_qk / tf.math.sqrt(dk)

  # add the mask to the scaled tensor.
  if mask is not None:
    scaled_attention_logits += (mask * -1e9)

  # softmax is normalized on the last axis (seq_len_k) so that the scores
  # add up to 1.
  attention_weights = tf.nn.softmax(scaled_attention_logits, axis=-1)  # (..., seq_len_q, seq_len_k)

  output = tf.matmul(attention_weights, v)  # (..., seq_len_q, depth_v)

  return output, attention_weights


class MultiHeadAttention(tf.keras.layers.Layer):
  def __init__(self, d_model, num_heads):
    super(MultiHeadAttention, self).__init__()
    self.num_heads = num_heads
    self.d_model = d_model

    assert d_model % self.num_heads == 0

    self.depth = d_model // self.num_heads

    self.wq = tf.keras.layers.Dense(d_model)
    self.wk = tf.keras.layers.Dense(d_model)
    self.wv = tf.keras.layers.Dense(d_model)

    self.dense = tf.keras.layers.Dense(d_model)

  def split_heads(self, x, batch_size):
    """Split the last dimension into (num_heads, depth).
    Transpose the result such that the shape is (batch_size, num_heads, seq_len, depth)
    """
    x = tf.reshape(x, (batch_size, -1, self.num_heads, self.depth))
    return tf.transpose(x, perm=[0, 2, 1, 3])

  def call(self, v, k, q, mask):
    batch_size = tf.shape(q)[0]

    q = self.wq(q)  # (batch_size, seq_len, d_model)
    k = self.wk(k)  # (batch_size, seq_len, d_model)
    v = self.wv(v)  # (batch_size, seq_len, d_model)

    q = self.split_heads(q, batch_size)  # (batch_size, num_heads, seq_len_q, depth)
    k = self.split_heads(k, batch_size)  # (batch_size, num_heads, seq_len_k, depth)
    v = self.split_heads(v, batch_size)  # (batch_size, num_heads, seq_len_v, depth)

    # scaled_attention.shape == (batch_size, num_heads, seq_len_q, depth)
    # attention_weights.shape == (batch_size, num_heads, seq_len_q, seq_len_k)
    scaled_attention, attention_weights = scaled_dot_product_attention(
        q, k, v, mask)

    scaled_attention = tf.transpose(scaled_attention, perm=[0, 2, 1, 3])  # (batch_size, seq_len_q, num_heads, depth)

    concat_attention = tf.reshape(scaled_attention,
                                  (batch_size, -1, self.d_model))  # (batch_size, seq_len_q, d_model)

    output = self.dense(concat_attention)  # (batch_size, seq_len_q, d_model)

    return output, attention_weights

  
def point_wise_feed_forward_network(d_model, dff):
  return tf.keras.Sequential([
      tf.keras.layers.Dense(dff, activation='relu'),  # (batch_size, seq_len, dff)
      tf.keras.layers.Dense(d_model)  # (batch_size, seq_len, d_model)
  ])


class EncoderLayer(tf.keras.layers.Layer):
  def __init__(self, d_model, num_heads, dff, rate=0.1):
    super(EncoderLayer, self).__init__()

    self.mha = MultiHeadAttention(d_model, num_heads)
    self.ffn = point_wise_feed_forward_network(d_model, dff)

    self.layernorm1 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
    self.layernorm2 = tf.keras.layers.LayerNormalization(epsilon=1e-6)

    self.dropout1 = tf.keras.layers.Dropout(rate)
    self.dropout2 = tf.keras.layers.Dropout(rate)

  def call(self, x, training, mask):

    attn_output, _ = self.mha(x, x, x, mask)  # (batch_size, input_seq_len, d_model)
    attn_output = self.dropout1(attn_output, training=training)
    out1 = self.layernorm1(x + attn_output)  # (batch_size, input_seq_len, d_model)

    ffn_output = self.ffn(out1)  # (batch_size, input_seq_len, d_model)
    ffn_output = self.dropout2(ffn_output, training=training)
    out2 = self.layernorm2(out1 + ffn_output)  # (batch_size, input_seq_len, d_model)

    return out2


class DecoderLayer(tf.keras.layers.Layer):
  def __init__(self, d_model, num_heads, dff, rate=0.1):
    super(DecoderLayer, self).__init__()

    self.mha1 = MultiHeadAttention(d_model, num_heads)
    self.mha2 = MultiHeadAttention(d_model, num_heads)

    self.ffn = point_wise_feed_forward_network(d_model, dff)

    self.layernorm1 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
    self.layernorm2 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
    self.layernorm3 = tf.keras.layers.LayerNormalization(epsilon=1e-6)

    self.dropout1 = tf.keras.layers.Dropout(rate)
    self.dropout2 = tf.keras.layers.Dropout(rate)
    self.dropout3 = tf.keras.layers.Dropout(rate)

  def call(self, x, enc_output, training,
           look_ahead_mask, padding_mask):
    # enc_output.shape == (batch_size, input_seq_len, d_model)

    attn1, attn_weights_block1 = self.mha1(x, x, x, look_ahead_mask)  # (batch_size, target_seq_len, d_model)
    attn1 = self.dropout1(attn1, training=training)
    out1 = self.layernorm1(attn1 + x)

    attn2, attn_weights_block2 = self.mha2(
        enc_output, enc_output, out1, padding_mask)  # (batch_size, target_seq_len, d_model)
    attn2 = self.dropout2(attn2, training=training)
    out2 = self.layernorm2(attn2 + out1)  # (batch_size, target_seq_len, d_model)

    ffn_output = self.ffn(out2)  # (batch_size, target_seq_len, d_model)
    ffn_output = self.dropout3(ffn_output, training=training)
    out3 = self.layernorm3(ffn_output + out2)  # (batch_size, target_seq_len, d_model)

    return out3, attn_weights_block1, attn_weights_block2

class Encoder(tf.keras.layers.Layer):
  def __init__(self, num_layers, d_model, num_heads, dff, input_vocab_size,
               maximum_position_encoding, rate=0.1):
    super(Encoder, self).__init__()

    self.d_model = d_model
    self.num_layers = num_layers

    self.embedding = tf.keras.layers.Embedding(input_vocab_size, d_model)
    self.pos_encoding = positional_encoding(maximum_position_encoding,
                                            self.d_model)

    self.enc_layers = [EncoderLayer(d_model, num_heads, dff, rate)
                       for _ in range(num_layers)]

    self.dropout = tf.keras.layers.Dropout(rate)

  def call(self, x, training, mask):

    seq_len = tf.shape(x)[1]

    # adding embedding and position encoding.
    x = self.embedding(x)  # (batch_size, input_seq_len, d_model)
    x *= tf.math.sqrt(tf.cast(self.d_model, tf.float32))
    x += self.pos_encoding[:, :seq_len, :]

    x = self.dropout(x, training=training)

    for i in range(self.num_layers):
      x = self.enc_layers[i](x, training, mask)

    return x  # (batch_size, input_seq_len, d_model)

class Decoder(tf.keras.layers.Layer):
  def __init__(self, num_layers, d_model, num_heads, dff, target_vocab_size,
               maximum_position_encoding, rate=0.1):
    super(Decoder, self).__init__()

    self.d_model = d_model
    self.num_layers = num_layers

    self.embedding = tf.keras.layers.Embedding(target_vocab_size, d_model)
    self.pos_encoding = positional_encoding(maximum_position_encoding, d_model)

    self.dec_layers = [DecoderLayer(d_model, num_heads, dff, rate)
                       for _ in range(num_layers)]
    self.dropout = tf.keras.layers.Dropout(rate)

  def call(self, x, enc_output, training,
           look_ahead_mask, padding_mask):

    seq_len = tf.shape(x)[1]
    attention_weights = {}

    x = self.embedding(x)  # (batch_size, target_seq_len, d_model)
    x *= tf.math.sqrt(tf.cast(self.d_model, tf.float32))
    x += self.pos_encoding[:, :seq_len, :]

    x = self.dropout(x, training=training)

    for i in range(self.num_layers):
      x, block1, block2 = self.dec_layers[i](x, enc_output, training,
                                             look_ahead_mask, padding_mask)

      attention_weights[f'decoder_layer{i+1}_block1'] = block1
      attention_weights[f'decoder_layer{i+1}_block2'] = block2

    # x.shape == (batch_size, target_seq_len, d_model)
    return x, attention_weights

  
train_loss = tf.keras.metrics.Mean(name='train_loss')
train_pos_loss = tf.keras.metrics.Mean(name='train_pos_loss')
train_neg_loss = tf.keras.metrics.Mean(name='train_neg_loss')
train_accuracy = tf.keras.metrics.Mean(name='train_accuracy')


class Transformer(tf.keras.Model):
  def __init__(self, num_layers, d_model, num_heads, dff, input_vocab_size,
               target_vocab_size, pe_input, pe_target, rate=0.1):
    super().__init__()
    self.encoder = Encoder(num_layers, d_model, num_heads, dff,
                             input_vocab_size, pe_input, rate)

    self.decoder = Decoder(num_layers, d_model, num_heads, dff,
                           target_vocab_size, pe_target, rate)

    self.final_layer = tf.keras.layers.Dense(target_vocab_size)

  def call(self, inputs, training):
    # Keras models prefer if you pass all your inputs in the first argument
    inp, tar = inputs

    enc_padding_mask, look_ahead_mask, dec_padding_mask = self.create_masks(inp, tar)

    enc_output = self.encoder(inp, training, enc_padding_mask)  # (batch_size, inp_seq_len, d_model)

    # dec_output.shape == (batch_size, tar_seq_len, d_model)
    dec_output, attention_weights = self.decoder(
        tar, enc_output, training, look_ahead_mask, dec_padding_mask)

    final_output = self.final_layer(dec_output)  # (batch_size, tar_seq_len, target_vocab_size)

    return final_output, attention_weights

  def create_masks(self, inp, tar):
    # Encoder padding mask
    enc_padding_mask = create_padding_mask(inp)

    # Used in the 2nd attention block in the decoder.
    # This padding mask is used to mask the encoder outputs.
    dec_padding_mask = create_padding_mask(inp)

    # Used in the 1st attention block in the decoder.
    # It is used to pad and mask future tokens in the input received by
    # the decoder.
    look_ahead_mask = create_look_ahead_mask(tf.shape(tar)[1])
    dec_target_padding_mask = create_padding_mask(tar)
    look_ahead_mask = tf.maximum(dec_target_padding_mask, look_ahead_mask)

    return enc_padding_mask, look_ahead_mask, dec_padding_mask


class CustomSchedule(tf.keras.optimizers.schedules.LearningRateSchedule):
  def __init__(self, d_model, warmup_steps):
    super(CustomSchedule, self).__init__()
    self.d_model = d_model
    self.d_model = tf.cast(self.d_model, tf.float32)
    self.warmup_steps = warmup_steps

  def __call__(self, step):
    arg1 = tf.math.rsqrt(step)
    arg2 = step * (self.warmup_steps ** -1.5)
    return tf.math.rsqrt(self.d_model) * tf.math.minimum(arg1, arg2)



loss_object = tf.keras.losses.SparseCategoricalCrossentropy(
    from_logits=True, reduction='none')

def entropy_loss(pred):
  # pred is (batch_size * seq_len * out_vocab_size)
  probs = tf.nn.softmax(pred)
  entropy = - tf.reduce_sum(probs * tf.math.log(probs), axis=-1)
  return entropy

def my_sampler(logits):
  sample = tf.random.categorical(logits, 1)
  sample = sample[:,0]
  return sample

def loss_for_negatives(real, pred):
  loss_ = loss_object(real, pred)
  loss_ = tf.maximum(NEG_CLIP, loss_)
  return - loss_

  # real_onehot = tf.one_hot(real, pred.shape[-1])
  # pred_masked_real = pred - real_onehot * 1e10

  # sampled_target = tf.map_fn(my_sampler, pred_masked_real, fn_output_signature=tf.TensorSpec(shape=[None], dtype=tf.int64))
  # loss_ = loss_object(sampled_target, pred)

  # for i in range(9):
  #   sampled_target = tf.map_fn(my_sampler, pred_masked_real, fn_output_signature=tf.TensorSpec(shape=[None], dtype=tf.int64))
  #   loss_ += loss_object(sampled_target, pred)

  # loss_ /= 10.0
  # # loss_ -= loss_object(real, pred)

  # # probs = tf.nn.softmax(pred)
  # # loss_ = real_onehot * tf.abs((1-real_onehot)-probs)
  # # loss_ = tf.reduce_sum(loss_, axis=2)
  # return loss_

# loss_type = 1: only positives
# loss_type = -1: only negatives
def loss_function_selective(real, pred, ispositive, loss_type):
  loss_ = loss_object(real, pred)

  mask = tf.math.logical_not(tf.math.equal(real, 0))
  mask = tf.cast(mask, dtype=loss_.dtype)

  loss_ *= mask
  loss = tf.reduce_sum(loss_, axis=-1)/tf.reduce_sum(mask, axis=-1)

  pos_loss = loss * ispositive * (loss_type + 1.0) / 2
  neg_loss = loss * (ispositive-1.0) * (loss_type - 1.0) / 2
  # loss = pos_loss + neg_loss

  loss = (1-ispositive) * loss

  return pos_loss, neg_loss, loss
  

def loss_function(real, pred, ispositive):
  pos_loss_ = loss_object(real, pred)

  mask = tf.math.logical_not(tf.math.equal(real, 0))
  mask = tf.cast(mask, dtype=pos_loss_.dtype)

  pos_loss_ *= mask
  pos_loss = tf.reduce_sum(pos_loss_, axis=-1)/tf.reduce_sum(mask, axis=-1)
  pos_loss = ispositive * pos_loss

  neg_loss_ = loss_for_negatives(real, pred)
  neg_loss_ *= mask
  # neg_loss = tf.reduce_sum(neg_loss_, axis=-1)/tf.reduce_sum(mask, axis=-1)
  neg_loss = tf.reduce_min(neg_loss_, axis=-1)
  neg_loss = (1-ispositive) * neg_loss

  
  # ent_loss_ = entropy_loss(pred)
  # ent_loss_ *= mask
  # ent_loss = tf.reduce_sum(ent_loss_, axis=-1)/tf.reduce_sum(mask, axis=-1)
  # neg_loss = tf.maximum(-NEG_CLIP, (ispositive - 1) * loss)
  
  loss = pos_loss + NEG_WEIGHT * neg_loss #+ ENT_WEIGHT * ent_loss
  loss = tf.reduce_mean(loss)
  pos_loss = tf.reduce_mean(pos_loss)
  neg_loss = tf.reduce_mean(neg_loss)
  return pos_loss, neg_loss, loss


def my_gather(x):
  return tf.gather(x[0], x[1])

def my_gather_list(x):
  return tf.map_fn(my_gather, x, fn_output_signature=tf.float32)

loss_object2 = tf.keras.losses.BinaryCrossentropy(
    from_logits=True, reduction='none')

def funny_loss_function(real, pred, ispositive):
  mask = tf.math.logical_not(tf.math.equal(real, 0))

  # binary crossentropy on the target logit
  target_logits = tf.map_fn(my_gather_list, (pred, real), fn_output_signature=tf.TensorSpec(shape=[None], dtype=tf.float32))
  target_logits = tf.expand_dims(target_logits, -1)
  loss_ = loss_object2(tf.ones_like(target_logits), target_logits)

  loss_ += loss_object(real, pred)

  # small force pulling all logits to the opposite direction
  reg_loss_ = tf.reduce_mean(pred, axis=-1)
  # loss_ = loss_ + 0.001 * reg_loss_
  
  mask = tf.cast(mask, dtype=loss_.dtype)
  loss_ *= mask

  loss = tf.reduce_sum(loss_, axis=-1)/tf.reduce_sum(mask, axis=-1)
  pos_loss = ispositive * loss
  neg_loss = tf.maximum(-NEG_CLIP, (ispositive - 1) * loss)
  loss = pos_loss + NEG_WEIGHT * neg_loss
  loss = tf.reduce_mean(loss)
  pos_loss = tf.reduce_mean(pos_loss)
  neg_loss = tf.reduce_mean(neg_loss)
  return pos_loss, neg_loss, loss


def accuracy_function(real, pred, ispositive):
  accuracies = tf.equal(real, tf.argmax(pred, axis=2))

  mask = tf.math.logical_not(tf.math.equal(real, 0))
  accuracies = tf.math.logical_and(mask, accuracies)

  accuracies = tf.cast(accuracies, dtype=tf.float32)
  mask = tf.cast(mask, dtype=tf.float32)
  return tf.reduce_sum(accuracies)/tf.reduce_sum(mask)


# The @tf.function trace-compiles train_step into a TF graph for faster
# execution. The function specializes to the precise shape of the argument
# tensors. To avoid re-tracing due to the variable sequence lengths or variable
# batch sizes (the last batch is smaller), use input_signature to specify
# more generic shapes.

train_step_signature = [
  tf.TensorSpec(shape=(None, None), dtype=tf.int64),
  tf.TensorSpec(shape=(None, None), dtype=tf.int64),
  tf.TensorSpec(shape=(None, ), dtype=tf.float32),
  tf.TensorSpec(shape=(), dtype=tf.float32),
]


@tf.function(input_signature=train_step_signature)
def train_step(inp, tar, ispositive, loss_type):
  tar_inp = tar[:, :-1]
  tar_real = tar[:, 1:]

  with tf.GradientTape() as tape:
    predictions, _ = transformer([inp, tar_inp],
                                 training = True)
    # pos_loss, neg_loss, loss = loss_function(tar_real, predictions, ispositive)
    pos_loss, neg_loss, loss = loss_function_selective(tar_real, predictions, ispositive, loss_type)
    # pos_loss, neg_loss, loss = tf.cond(tf.equal(loss_type, 0.0),
    #                                    lambda: loss_function(tar_real, predictions, ispositive),
    #                                    lambda: loss_function_selective(tar_real, predictions, ispositive, loss_type)
    #                                    )

  gradients = tape.gradient(loss, transformer.trainable_variables)
  gradients = [tf.clip_by_norm(g, CLIP_NORM) for g in gradients]
  optimizer.apply_gradients(zip(gradients, transformer.trainable_variables))

  train_loss(loss)
  train_pos_loss(pos_loss)
  train_neg_loss(neg_loss)
  train_accuracy(accuracy_function(tar_real, predictions, ispositive))

def train(epochs, transf, optim, train_batches, loss_type, ckpt_manager=None):
  global transformer, optimizer
  transformer = transf
  optimizer = optim
  for epoch in range(epochs):
    start = time.time()
    train_loss.reset_states()
    train_pos_loss.reset_states()
    train_neg_loss.reset_states()
    train_accuracy.reset_states()

    for (batch, (inp, tar, ispositive)) in enumerate(train_batches):
      train_step(inp, tar, ispositive, loss_type)

      if batch % 50 == 0:
        print(f'Epoch {epoch + 1} Batch {batch} Loss {train_loss.result():.4f}/{train_pos_loss.result():.4f}/{train_neg_loss.result():.4f} Accuracy {train_accuracy.result():.4f}')

      if (ckpt_manager is not None) and ((epoch + 1) % 5 == 0):
        ckpt_save_path = ckpt_manager.save()
        print(f'Saving checkpoint for epoch {epoch+1} at {ckpt_save_path}')

    print(f'Epoch {epoch + 1}: Time {time.time() - start:.2f} secs, Loss {train_loss.result():.4f}/{train_pos_loss.result():.4f}/{train_neg_loss.result():.4f}, Accuracy {train_accuracy.result():.4f}')


class Translator(tf.Module):
  def __init__(self, tokenizer_in, tokenizer_out, transf):
    self.tokenizer_in = tokenizer_in
    self.tokenizer_out = tokenizer_out
    self.transformer = transf
    self.vocab_out = self.tokenizer_out.get_vocabulary()

    start_end = self.tokenizer_out(['SOS EOS'])[0] # TODO this may have to be parameterised
    self.start = start_end[0][tf.newaxis]
    self.end = start_end[1][tf.newaxis]
    
  def __call__(self, sentence, max_length=20, deterministic=False):
    assert isinstance(sentence, tf.Tensor)
    if len(sentence.shape) == 0:
      sentence = sentence[tf.newaxis]

    sentence = self.tokenizer_in(sentence)
    encoder_input = sentence

    # `tf.TensorArray` is required here (instead of a python list) so that the
    # dynamic-loop can be traced by `tf.function`.
    output_array = tf.TensorArray(dtype=tf.int64, size=0, dynamic_size=True)
    output_array = output_array.write(0, self.start)

    probs = []

    for i in tf.range(max_length):
      output = tf.transpose(output_array.stack())
      predictions, _ = self.transformer([encoder_input, output], training=False)

      # select the last token from the seq_len dimension
      predictions = predictions[:, -1:, :]  # (batch_size, 1, vocab_size)

      if deterministic:
        predicted_id = tf.argmax(predictions, axis=-1)
      else:
        predicted_id = tf.random.categorical(predictions[0], 1)
      
      predicted_probs = tf.nn.softmax(predictions)[0][0]
      predicted_prob = tf.gather(predicted_probs, [predicted_id])
      probs.append(predicted_prob[0][0][0].numpy())
      # probs = predicted_probs.numpy()
      # print("\n\npred_id: ", predicted_id[0], predicted_prob)
      # for j, p in enumerate(probs):
      #   print(" {} -> {}".format(self.vocab_out[j], p))

      # concatentate the predicted_id to the output which is given to the decoder
      # as its input.
      output_array = output_array.write(i+1, predicted_id[0])

      if predicted_id == self.end:
        break

    output = tf.transpose(output_array.stack())
    # print("outputs: ", output)
    # print("probs  : ", probs)
    # output.shape (1, tokens)
    # text = tokenizers.en.detokenize(output)[0]  # shape: ()
    tokens = tf.gather(self.vocab_out, output)
    return tokens, probs


  def beamsearch(self, sentence, parse=True, max_length=20, beamsize=10):
    assert isinstance(sentence, tf.Tensor)
    if len(sentence.shape) == 0:
      sentence = sentence[tf.newaxis]

    sentence = self.tokenizer_in(sentence)
    encoder_input = sentence

    output_array = [self.start[0]]

    # list of top k output sequences
    top = [(1.0, output_array, 1, False)]

    while len(top) > 0:  
      # expand most probable sequence that hasn't ended yet

      index = 0
      found = False
      while index < len(top):
        p_curr, output_curr, len_curr, ended_curr = top[index]
        if not ended_curr:
          del top[index]
          found = True
          break
        index += 1
        
      if not found: # no more sequences to expand
        break

      output = tf.convert_to_tensor([output_curr])
      predictions, _ = self.transformer([encoder_input, output], training=False)
      predictions = predictions[:, -1:, :]  # (batch_size, 1, vocab_size)      
      predicted_probs = tf.nn.softmax(predictions)[0][0]

      # tokens = tf.gather(self.vocab_out, output)
      # tokens = tokens[0].numpy()
      # tokens = [t.decode('UTF-8') for t in tokens]
      # print("----", p_curr, " ".join(tokens))

      cumulative_probs = predicted_probs * p_curr
      # cumulative_probs = tf.minimum(predicted_probs, p_curr)

      # get the k best continuations
      k = tf.minimum(beamsize, cumulative_probs.shape[0])
      selected_probs = tf.math.top_k(cumulative_probs, sorted=True, k=k)

      # merge them into top
      values = tf.unstack(selected_probs.values)
      indices = tf.unstack(tf.cast(selected_probs.indices, dtype=tf.int64))
      start_index = 0
      for prob, predicted_id in zip(values, indices):
        if start_index == len(top) and len(top) >= beamsize:
          break

        new_output = copy.deepcopy(output_curr)
        new_output.append(predicted_id)
        while(True):
          if start_index >= len(top):
            break
          else:
            top_prob, _, _, _ = top[start_index]
            if top_prob < prob:            
              break
          start_index += 1
        end = tf.constant(predicted_id == self.end[0] or len_curr+1 >= max_length)
        top.insert(start_index, (prob, new_output, len_curr+1, end))
      top = top[:beamsize]

    result = []
    for t in top:
      prob = t[0]
      output_array = t[1]
      output = tf.convert_to_tensor([output_array])
      tokens = tf.gather(self.vocab_out, output)
      tokens = tokens[0].numpy()
      tokens = [t.decode('UTF-8') for t in tokens]
      if parse:
        rule, isvalid = parse_rule(tokens)
        if isvalid:
          result.append((prob, rule))
      else:
        result.append((prob, " ".join(tokens)))

    return result

  def beamsearch_with_critique(self, sentence, critique, parse=True, max_length=20, beamsize=10):
    assert isinstance(sentence, tf.Tensor)
    if len(sentence.shape) == 0:
      sentence = sentence[tf.newaxis]

    sentence = self.tokenizer_in(sentence)
    encoder_input = sentence

    output_array = [self.start[0]]

    # list of top k output sequences
    top = [(1.0, 1.0, output_array, 1, False)]

    while len(top) > 0:  
      # expand most probable sequence that hasn't ended yet

      index = 0
      found = False
      while index < len(top):
        p_curr, p_curr_c, output_curr, len_curr, ended_curr = top[index]
        if not ended_curr:
          del top[index]
          found = True
          break
        index += 1
        
      if not found: # no more sequences to expand
        break

      output = tf.convert_to_tensor([output_curr])
      predictions, _ = self.transformer([encoder_input, output], training=False)
      predictions = predictions[:, -1:, :]  # (batch_size, 1, vocab_size)      
      predicted_probs = tf.nn.softmax(predictions)[0][0]

      predictions_c, _ = critique([encoder_input, output], training=False)
      predictions_c = predictions_c[:, -1:, :]  # (batch_size, 1, vocab_size)      
      predicted_probs_c = tf.nn.softmax(predictions_c)[0][0]

      # print(predicted_probs)
      # print(predicted_probs_c)
      # print("-----")

      # tokens = tf.gather(self.vocab_out, output)
      # tokens = tokens[0].numpy()
      # tokens = [t.decode('UTF-8') for t in tokens]
      # print("----", p_curr, " ".join(tokens))

      cumulative_probs = predicted_probs * p_curr
      cumulative_probs_c = predicted_probs_c * p_curr_c

      # get the k best continuations
      k = tf.minimum(beamsize, cumulative_probs.shape[0])
      selected_probs = tf.math.top_k(cumulative_probs, sorted=True, k=k)

      # merge them into top
      values = tf.unstack(selected_probs.values)
      indices = tf.unstack(tf.cast(selected_probs.indices, dtype=tf.int64))
      values_c = tf.gather(cumulative_probs_c, indices)
      start_index = 0
      for prob, prob_c, predicted_id in zip(values, values_c, indices):
        if start_index == len(top) and len(top) >= beamsize:
          break

        new_output = copy.deepcopy(output_curr)
        new_output.append(predicted_id)
        while(True):
          if start_index >= len(top):
            break
          else:
            top_prob, _, _, _, _ = top[start_index]
            if top_prob < prob:            
              break
          start_index += 1
        end = tf.constant(predicted_id == self.end[0] or len_curr+1 >= max_length)
        top.insert(start_index, (prob, prob_c, new_output, len_curr+1, end))
      top = top[:beamsize]

    result = []
    for t in top:
      prob = t[0]
      prob_c = t[1]
      output_array = t[2]
      output = tf.convert_to_tensor([output_array])
      tokens = tf.gather(self.vocab_out, output)
      tokens = tokens[0].numpy()
      tokens = [t.decode('UTF-8') for t in tokens]
      if parse:
        rule, isvalid = parse_rule(tokens)
        if isvalid:
          result.append((prob, prob_c, rule))
      else:
        result.append((prob, prob_c, " ".join(tokens)))

    return result

def parse_rule(tokens):
  if tokens[0] != 'SOS':
    return -1, False
  if tokens[-1] != 'EOS':
    return -1, False
  tokens = tokens[1:-1]

  eoh_count = tokens.count('EOH')
  if eoh_count != 1:
    return -1, False
  
  eoh = tokens.index('EOH')
  head = tokens[:eoh]
  body = tokens[eoh+1:]
  atoms = [list(y) for x, y in itertools.groupby(body, lambda z: z == 'EOP') if not x]

  atoms.append(head)
  formatted_atoms = []
  for a in atoms:
    if len(head) > 3 or len(head) < 2:
      return -1, False
    a2 = "{}({})".format(a[0], ",".join(a[1:]))
    formatted_atoms.append(a2)

  rule = "{} -> {}".format(", ".join(formatted_atoms[:-1]), formatted_atoms[-1])  
  return rule, True
