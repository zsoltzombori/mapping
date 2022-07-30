# # divide GPUs, by randomly selecting one for each process
# import os
# import numpy as np
# gpus = os.environ["CUDA_VISIBLE_DEVICES"].split(',')
# gpu_count = len(gpus)
# gpu = gpus[np.random.randint(0, gpu_count)]
# os.environ["CUDA_VISIBLE_DEVICES"] = gpu

import tensorflow as tf
print("GPU available: ", tf.config.list_physical_devices('GPU'))
physical_devices = tf.config.list_physical_devices('GPU')
try:
    tf.config.experimental.set_memory_growth(physical_devices[0], True)
except:
  # Invalid device or cannot modify virtual devices once initialized.
  pass

import logging
logging.getLogger('tensorflow').setLevel(logging.ERROR)  # suppress warnings

import time
import sys
import argparse
import re

import transformer

parser = argparse.ArgumentParser()
parser.add_argument('--datadir', type=str, required=True)
parser.add_argument('--epochs', type=int, default=100)
parser.add_argument('--batch_size', type=int, default=20)
parser.add_argument('--neg_weight', type=float, default=5.0)
parser.add_argument('--beamsize', type=int, default=10)
parser.add_argument('--num_layers', type=str, default="3")
parser.add_argument('--d_model', type=int, default=256)
parser.add_argument('--checkpoint_path', type=str, default="none")
parser.add_argument('--lr', type=float, default=0.001)
parser.add_argument('--lr_decay_steps', type=int, default=1000)
parser.add_argument('--optimizer', type=str, default="sgd")
parser.add_argument('--beta1', type=float, default=0.5)
parser.add_argument('--beta2', type=float, default=0.999)
parser.add_argument('--char_tokenizer', type=int, default=0)
parser.add_argument('--remove_args', type=int, default=0)
parser.add_argument('--loss_type', type=str, default="lprp") #"nll", "prp", "lprp", "joint_lprp"
parser.add_argument('--split', type=str, default="0.7,0.15,0.15")
parser.add_argument('--outdir', type=str)
parser.add_argument('--monitor_probs', type=int, default=0)
parser.add_argument('--filter_pn', type=int, default=0)
parser.add_argument('--seed', type=int, default=100)
parser.add_argument('--seq_out_len', type=int, default=20)

args = parser.parse_args()

for arg in vars(args):
  print(arg, getattr(args, arg))

DATADIR = args.datadir
EPOCHS = args.epochs
BATCH_SIZE=args.batch_size
NEG_WEIGHT=args.neg_weight
BEAMSIZE=args.beamsize
LOSS_TYPE=args.loss_type
SPLIT=[float(x) for x in args.split.split(",")]
MONITOR_PROBS = args.monitor_probs == 1
FILTER_PN = args.filter_pn == 1
OUTDIR = args.outdir

# tokenizer parameters
CHAR_TOKENIZER = args.char_tokenizer == 1
MAX_VOCAB_SIZE_IN = 10000
MAX_VOCAB_SIZE_OUT = 10000
MAX_SEQUENCE_LENGTH_IN = 20
MAX_SEQUENCE_LENGTH_OUT = args.seq_out_len


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
CHECKPOINT_PATH= args.checkpoint_path
if CHECKPOINT_PATH.lower() == "none":
  CHECKPOINT_PATH = None
BUFFER_SIZE = 200000
REMOVE_ARGS = args.remove_args == 1

tf.random.set_seed(args.seed)


# load data
examples, max_input_len_w, max_input_len_c, max_output_len_w, max_output_len_c = transformer.load_data(DATADIR, BUFFER_SIZE, split=SPLIT)


if CHAR_TOKENIZER:
  MAX_SEQUENCE_LENGTH_IN = max_input_len_c
  MAX_SEQUENCE_LENGTH_OUT = max_output_len_e
else:
  MAX_SEQUENCE_LENGTH_IN = max_input_len_w
  MAX_SEQUENCE_LENGTH_OUT = max_output_len_w

# create vectorizers
first=True
for example_type in examples:
    examples_train = examples[example_type][0]
    text_in_curr = examples_train.map(lambda x: x["input"])
    text_out_curr = examples_train.map(lambda x: x["output"])
    if first:
        text_in = text_in_curr
        text_out = text_out_curr
        first=False
    else:
        text_in = text_in.concatenate(text_in_curr)
        text_out = text_out.concatenate(text_out_curr)

tokenizer_in = transformer.MyTokenizer(text_in, MAX_VOCAB_SIZE_IN, MAX_SEQUENCE_LENGTH_IN, CHAR_TOKENIZER)
tokenizer_out = transformer.MyTokenizer(text_out, MAX_VOCAB_SIZE_OUT, MAX_SEQUENCE_LENGTH_OUT, CHAR_TOKENIZER)
  
print("Input vocab size: ", len(tokenizer_in.vocabulary))
# print(tokenizer_in.vocabulary)
print("Output vocab size: ", len(tokenizer_out.vocabulary))
# print(tokenizer_out.vocabulary)


# create batches of training data
if "pos" in examples:
  pos_examples, pos_examples_val, pos_examples_test = examples["pos"]
  pos_size = transformer.count_dataset(pos_examples)
  BATCH_SIZE = min(BATCH_SIZE, pos_size)
else:
  assert False, "MISSING POSITIVE SUPERVISION!!!"

if ("neg" not in examples) or (NEG_WEIGHT == 0):
    neg_batches = None
    neg_examples = None
else:
    neg_examples, neg_examples_val, neg_examples_test = examples["neg"]
    neg_size = transformer.count_dataset(neg_examples)
    BATCH_SIZE = min(BATCH_SIZE, neg_size)
    neg_batches = transformer.make_batches(neg_examples, tokenizer_in, tokenizer_out, BUFFER_SIZE, BATCH_SIZE, REMOVE_ARGS)

  
pos_batches = transformer.make_batches(pos_examples, tokenizer_in, tokenizer_out, BUFFER_SIZE, BATCH_SIZE, REMOVE_ARGS)


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
  # optimizer = tf.keras.optimizers.Adam(learning_rate, beta_1=0.9, beta_2=0.98, epsilon=1e-9)
  optimizer = tf.keras.optimizers.Adam(learning_rate, beta_1=BETA_1, beta_2=BETA_2, epsilon=1e-7)
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
  # input_vocab_size=MAX_VOCAB_SIZE_IN, target_vocab_size=MAX_VOCAB_SIZE_OUT,
  input_vocab_size=vocab_size_in, target_vocab_size=vocab_size_out,
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
transformer.train(EPOCHS, my_transformer, optimizer, pos_batches, neg_batches, NEG_WEIGHT, LOSS_TYPE,
                  outdir=OUTDIR,
                  monitor_probs=MONITOR_PROBS,
                  filter_pn=FILTER_PN,
                  ckpt_manager=ckpt_manager)

if EPOCHS > 0:
    my_transformer.summary()

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


def safe_rule(rule, neg_examples):
  if neg_examples is None:
    return True
  for e in neg_examples:
    candidates = [c.numpy().decode("utf-8") for c in e["output"]]
    if rule in candidates:
      return False
  return True

def eval_beamsearch(translator, pos_e, neg_e, beamsize, max_length, remove_args):
  t0 = time.time()
  threshold = 0.5

  top1_pos = 0
  top5_pos = 0
  top10_pos = 0
  top1_neg = 0
  top5_neg = 0
  top10_neg = 0
  
  pos_success = 0
  neg_failure = 0
  count = 0
  for e in pos_e:
    count += 1
    sentence = e["input"]
    if remove_args:
      sentence = transformer.remove_input_arguments(sentence)
    translations = translator.beamsearch(tf.constant(sentence), beamsize=beamsize, max_length=max_length)
    candidates = [c.numpy().decode("utf-8") for c in e["output"]]
    candidates = [re.sub(' +', ' ', c) for c in candidates]
    
    pos_prob = 0.0
    neg_prob = 0.0
    firstrule=None
    firstprob=0.0
    t1p, t5p, t10p, t1n, t5n, t10n = 0, 0, 0, 0, 0, 0
    for i, (prob, text, isvalid, rule) in enumerate(translations):
      # if not isvalid:
      #   continue

      # print("xxx", prob, text)
      
      if firstrule is None:
        firstrule = text
        firstprob = prob
        
      if text in candidates:
        pos_prob += prob
        if i == 0:
          t1p = 1
        if i < 5:
          t5p = 1
        if i < 10:
          t10p = 1
          
          
      if not safe_rule(text, neg_e):
        neg_prob += prob
        if i == 0:
          t1n = 1
        if i < 5:
          t5n = 1
        if i < 10:
          t10n = 1

    top1_pos += t1p
    top5_pos += t5p
    top10_pos += t10p
    top1_neg += t1n
    top5_neg += t5n
    top10_neg += t10n
    
    failure = False
    if t1p == 1:
      pos_success += 1
    else:
      failure = True
    if t1n == 1:
      neg_failure += 1
      failure = True

    if ((count+1) % 10) == 0:
      print("\n", count)
      print("Positive top1: {}, top5: {}, top10: {}".format(top1_pos / count, top5_pos / count, top10_pos / count))
      print("Negative top1: {}, top5: {}, top10: {}".format(top1_neg / count, top5_neg / count, top10_neg / count))

    print(count)
    if False: #failure:
      print("---------FAILURE----------")
      print(f'{"Input:":15s}: {sentence.numpy()}')
      for o in e["output"]:
        print(f'{"   Output:":15s}: {o.numpy()}')
      print(f'{firstprob} {firstrule}')
      print(f'Pos prob: {pos_prob:.3f}, Neg prob: {neg_prob:.3f}')
    sys.stdout.flush()

  t1 = time.time()
  if count == 0:
    count = 1
  print("Evaltime: {:.2f} sec, positive success ratio: {}, negative failure ratio: {}".format(t1-t0, pos_success / count, neg_failure / count))
  print("Positive top1: {}, top5: {}, top10: {}".format(top1_pos / count, top5_pos / count, top10_pos / count))
  print("Negative top1: {}, top5: {}, top10: {}".format(top1_neg / count, top5_neg / count, top10_neg / count))


# print("\n\nEVALUATION on the validation set")
# if neg_examples is not None:
#   neg_examples_val = neg_examples.concatenate(neg_examples_val)
# else:
#   neg_examples_val = None  
# eval_beamsearch(my_translator, pos_examples_val, neg_examples_val, beamsize=BEAMSIZE, max_length=MAX_SEQUENCE_LENGTH_OUT, remove_args=REMOVE_ARGS)

print("\n\nEVALUATION on the train set")
eval_beamsearch(my_translator, pos_examples, neg_examples, beamsize=BEAMSIZE, max_length=MAX_SEQUENCE_LENGTH_OUT, remove_args=REMOVE_ARGS)
