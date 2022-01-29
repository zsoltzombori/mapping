# # divide GPUs, by randomly selecting one for each process
# import os
# import numpy as np
# gpus = os.environ["CUDA_VISIBLE_DEVICES"].split(',')
# gpu_count = len(gpus)
# gpu = gpus[np.random.randint(0, gpu_count)]
# os.environ["CUDA_VISIBLE_DEVICES"] = gpu


import time
import sys
import transformer
import tensorflow as tf
import argparse
parser = argparse.ArgumentParser()
parser.add_argument('--datadir', type=str, required=True)
parser.add_argument('--epochs', type=int, default=100)
parser.add_argument('--batch_size', type=int, default=20)
parser.add_argument('--neg_weight', type=float, default=5.0)
parser.add_argument('--beamsize', type=int, default=10)
parser.add_argument('--num_layers', type=int, default=3)
parser.add_argument('--d_model', type=int, default=256)
parser.add_argument('--checkpoint_path', type=str, default=None)
parser.add_argument('--lr', type=float, default=0.001)
parser.add_argument('--lr_decay_steps', type=int, default=1000)
parser.add_argument('--optimizer', type=str, default="sgd")
parser.add_argument('--beta1', type=float, default=0.5)
parser.add_argument('--beta2', type=float, default=0.999)
parser.add_argument('--char_tokenizer', type=int, default=0)
parser.add_argument('--remove_args', type=int, default=0)


args = parser.parse_args()

for arg in vars(args):
  print(arg, getattr(args, arg))

DATADIR = args.datadir
EPOCHS = args.epochs
BATCH_SIZE=args.batch_size
NEG_WEIGHT=args.neg_weight
BEAMSIZE=args.beamsize

# tokenizer parameters
CHAR_TOKENIZER = args.char_tokenizer == 1
MAX_VOCAB_SIZE_IN = 10000
MAX_VOCAB_SIZE_OUT = 10000
MAX_SEQUENCE_LENGTH_IN = 20
MAX_SEQUENCE_LENGTH_OUT = 20


# optimizer parameters
LR_TYPE="decay" #"custom"/"decay"/"plain"
LR = args.lr
DECAY_STEPS = args.lr_decay_steps
WARMUP_STEPS = 4000 # int(train_size / BATCH_SIZE * EPOCHS / 10)
OPTIMIZER=args.optimizer
BETA_1=args.beta1
BETA_2=args.beta2

# transformer parameters
NUM_LAYERS = args.num_layers
D_MODEL = args.d_model
DFF = 512
NUM_HEADS = 8
DROPOUT_RATE = 0.1

# other
CHECKPOINT_PATH=args.checkpoint_path
BUFFER_SIZE = 200000
REMOVE_ARGS = args.remove_args == 1

tf.random.set_seed(1000)

# load data
(pos_examples, neg_examples), max_input_len_w, max_input_len_c, max_output_len_w, max_output_len_c = transformer.load_data(DATADIR, BUFFER_SIZE)
pos_examples, pos_examples_val, pos_examples_test = pos_examples
neg_examples, neg_examples_val, neg_examples_test = neg_examples

# pos_examples = pos_examples.take(1)
# BATCH_SIZE = 1


if CHAR_TOKENIZER:
  MAX_SEQUENCE_LENGTH_IN = max_input_len_c
  MAX_SEQUENCE_LENGTH_OUT = max_output_len_e
else:
  MAX_SEQUENCE_LENGTH_IN = max_input_len_w
  MAX_SEQUENCE_LENGTH_OUT = max_output_len_w

# create vectorizers
pos_text_in = pos_examples.map(lambda x: x["input"])
pos_text_out = pos_examples.map(lambda x: x["output"])
neg_text_in = neg_examples.map(lambda x: x["input"])
neg_text_out = neg_examples.map(lambda x: x["output"])
text_in = pos_text_in.concatenate(neg_text_in)
text_out = pos_text_out.concatenate(neg_text_out)  
tokenizer_in = transformer.MyTokenizer(text_in, MAX_VOCAB_SIZE_IN, MAX_SEQUENCE_LENGTH_IN, CHAR_TOKENIZER)
tokenizer_out = transformer.MyTokenizer(text_out, MAX_VOCAB_SIZE_OUT, MAX_SEQUENCE_LENGTH_OUT, CHAR_TOKENIZER)
  
print("Input vocab size: ", len(tokenizer_in.vocabulary))
# print(tokenizer_in.vocabulary)
print("Output vocab size: ", len(tokenizer_out.vocabulary))
# print(tokenizer_out.vocabulary)

# create batches of training data
pos_size = tf.data.experimental.cardinality(pos_examples).numpy()
neg_size = tf.data.experimental.cardinality(neg_examples).numpy()
BATCH_SIZE = min(BATCH_SIZE, pos_size, neg_size)
pos_batches = transformer.make_batches(pos_examples, tokenizer_in, tokenizer_out, BUFFER_SIZE, BATCH_SIZE, REMOVE_ARGS)
neg_batches = transformer.make_batches(neg_examples, tokenizer_in, tokenizer_out, BUFFER_SIZE, BATCH_SIZE, REMOVE_ARGS)


# create optimizer
if LR_TYPE == "custom":
  learning_rate = CustomSchedule(D_MODEL, WARMUP_STEPS)
elif LR_TYPE == "decay":
  learning_rate = tf.keras.optimizers.schedules.ExponentialDecay(LR, decay_steps=DECAY_STEPS, decay_rate=0.9, staircase=True)
else:
  learning_rate = LR

if OPTIMIZER == "sgd":
  optimizer = tf.keras.optimizers.SGD(learning_rate=learning_rate, momentum=BETA_1, nesterov=False)
elif OPTIMIZER == "adam":
  #optimizer = tf.keras.optimizers.Adam(learning_rate, beta_1=0.9, beta_2=0.98, epsilon=1e-9)
  optimizer = tf.keras.optimizers.Adam(0.001, beta_1=BETA_1, beta_2=BETA_2, epsilon=1e-7)
elif OPTIMIZER == "nadam":
  optimizer = tf.keras.optimizers.Nadam(LR, beta_1=BETA_1, beta_2=BETA_2, epsilon=1e-7)
elif OPTIMIZER == "adamax":
  optimizer = tf.keras.optimizers.Adamax(LR, beta_1=BETA_1, beta_2=BETA_2, epsilon=1e-7)

# create transformer
vocab_size_in = len(tokenizer_in.vocabulary)
vocab_size_out = len(tokenizer_out.vocabulary)
my_transformer = transformer.Transformer(
  num_layers=NUM_LAYERS, d_model=D_MODEL,
  num_heads=NUM_HEADS, dff=DFF,
  input_vocab_size=MAX_VOCAB_SIZE_IN, target_vocab_size=MAX_VOCAB_SIZE_OUT,
  pe_input=1000, pe_target=1000, rate=DROPOUT_RATE)


if CHECKPOINT_PATH is not None:
  ckpt = tf.train.Checkpoint(transformer=my_transformer, optimizer=optimizer)
  ckpt_manager = tf.train.CheckpointManager(ckpt, CHECKPOINT_PATH, max_to_keep=5)
  # if a checkpoint exists, restore the latest checkpoint.
  if ckpt_manager.latest_checkpoint:
    ckpt.restore(ckpt_manager.latest_checkpoint)
    print('Latest checkpoint restored!!')
else:
  ckpt_manager = None


# train the transformer
transformer.train(EPOCHS, my_transformer, optimizer, pos_batches, neg_batches, NEG_WEIGHT, ckpt_manager=ckpt_manager)

# create a translator
my_translator = transformer.Translator(tokenizer_in, tokenizer_out, my_transformer)

def print_translation(sentence, pred_tokens, ground_truth, ispositive):
  cnt = tf.size(pred_tokens).numpy()
  text = tf.strings.reduce_join(pred_tokens, separator=' ').numpy()

  print("---------")
  print(f'{"Input:":15s}: {sentence.numpy()}')
  print(f'{"Prediction":15s}: {text}')
  print(f'{"Pred length":15s}: {cnt}')
  print(f'{"Ground truth":15s}: {ground_truth}')
  print(f'{"Positive":15s}: {ispositive.numpy()}')


# def eval(examples, iterations=10):
#   inputs = []
#   for e in examples:
#     sentence = e["input"]
#     if sentence in inputs:
#       continue
#     else:
#       inputs.append(sentence)

#     print("---------")
#     print(f'{"Input:":15s}: {sentence.numpy()}')
#     outputs = []
#     for i in range(iterations):
#       if i==0:
#         deterministic=True
#       else:
#         deterministic=False
#       pred_tokens, probs = my_translator(tf.constant(sentence), deterministic=deterministic)
#       prob = np.prod(probs)
#       text = tf.strings.reduce_join(pred_tokens, separator=' ').numpy()
#       outputs.append((prob, text))
#     outputs = sorted(outputs, reverse=True)
#     for (prob, text) in outputs:
#       print(f'{prob:.10f}: {text}')


def safe_rule(rule, neg_examples):
  for e in neg_examples:
    candidates = [c.numpy().decode("utf-8") for c in e["output"]]
    return not rule in candidates

def eval_beamsearch(translator, pos_examples, neg_examples, beamsize, max_length, remove_args):
  t0 = time.time()
  threshold = 0.5
  
  pos_success = 0
  neg_failure = 0
  count = 0
  for e in pos_examples:
    count += 1
    sentence = e["input"]
    if remove_args:
      sentence = transformer.remove_input_arguments(sentence)
    translations = translator.beamsearch(tf.constant(sentence), beamsize=beamsize, max_length=max_length)
    candidates = [c.numpy().decode("utf-8") for c in e["output"]]
    
    pos_prob = 0.0
    neg_prob = 0.0
    firstrule=None
    firstprob=0.0
    for (prob, text, isvalid, rule) in translations:
      if isvalid:
        if firstrule is None:
          firstrule = text
          firstprob = prob
        if text in candidates:
          pos_prob += prob
        if not safe_rule(text, neg_examples):
          neg_prob += prob

    failure = False
    if pos_prob > 1-threshold:
      pos_success += 1
    else:
      failure = True
    if neg_prob > threshold:
      neg_failure += 1
      failure = True

    print(count)
    if failure:
      print("---------FAILURE----------")
      print(f'{"Input:":15s}: {sentence.numpy()}')
      for o in e["output"]:
        print(f'{"   Output:":15s}: {o.numpy()}')
      print(f'{firstprob} {firstrule}')
      print(f'Pos prob: {pos_prob:.3f}, Neg prob: {neg_prob:.3f}')
    sys.stdout.flush()

  t1 = time.time()
  print("Evaltime: {:.2f} sec, positive success ratio: {}, negative failure ratio: {}".format(t1-t0, pos_success / count, neg_failure / count))

print("\n\nEVALUATION on the train set")
eval_beamsearch(my_translator, pos_examples, neg_examples, beamsize=BEAMSIZE, max_length=MAX_SEQUENCE_LENGTH_OUT, remove_args=REMOVE_ARGS)

print("\n\nEVALUATION on the validation set")
neg_examples_val = neg_examples.concatenate(neg_examples_val)
eval_beamsearch(my_translator, pos_examples_val, neg_examples_val, beamsize=BEAMSIZE, max_length=MAX_SEQUENCE_LENGTH_OUT, remove_args=REMOVE_ARGS)

