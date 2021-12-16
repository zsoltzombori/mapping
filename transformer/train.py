PARSE=True
MIN_POSNEG_RATIO = 10

# # divide GPUs, by randomly selecting one for each process
# import os
# import numpy as np
# gpus = os.environ["CUDA_VISIBLE_DEVICES"].split(',')
# gpu_count = len(gpus)
# gpu = gpus[np.random.randint(0, gpu_count)]
# os.environ["CUDA_VISIBLE_DEVICES"] = gpu


import transformer
import tensorflow as tf
import argparse
parser = argparse.ArgumentParser()
parser.add_argument('--datadir', type=str, required=True)
parser.add_argument('--epochs', type=int, default=100)
parser.add_argument('--batch_size', type=int, default=20)
parser.add_argument('--neg_weight', type=float, default=5.0)
parser.add_argument('--beamsize', type=int, default=30)
args = parser.parse_args()

DATADIR = args.datadir
EPOCHS = args.epochs
BATCH_SIZE=args.batch_size
NEG_WEIGHT=args.neg_weight
BEAMSIZE=args.beamsize

# tokenizer parameters
MAX_VOCAB_SIZE_IN = 200000
MAX_VOCAB_SIZE_OUT = 200000
MAX_SEQUENCE_LENGTH_IN = 20
MAX_SEQUENCE_LENGTH_OUT = 20

# optimizer parameters
LR_TYPE="decay" #"custom"/"decay"/"plain"
LR = 0.001
DECAY_STEPS = 1000
WARMUP_STEPS = 4000 # int(train_size / BATCH_SIZE * EPOCHS / 10)

# transformer parameters
NUM_LAYERS = 3
D_MODEL = 256
DFF = 512
NUM_HEADS = 8
DROPOUT_RATE = 0.1

# other
USE_CHECKPOINT=False
BUFFER_SIZE = 200000
MAX_EVAL_LENGTH = 20

# load data
(pos_examples, neg_examples) = transformer.load_data(DATADIR, BUFFER_SIZE)
# pos_examples = pos_examples.take(1)
# neg_examples = neg_examples.take(1)

# create vectorizers
pos_text_in = pos_examples.map(lambda x: x["input"])
pos_text_out = pos_examples.map(lambda x: x["output"])
neg_text_in = neg_examples.map(lambda x: x["input"])
neg_text_out = neg_examples.map(lambda x: x["output"])
text_in = pos_text_in.concatenate(neg_text_in)
text_out = pos_text_out.concatenate(neg_text_out)


tokenizer_in = transformer.create_tokenizer(MAX_VOCAB_SIZE_IN, MAX_SEQUENCE_LENGTH_IN, text_in)
tokenizer_out = transformer.create_tokenizer(MAX_VOCAB_SIZE_OUT, MAX_SEQUENCE_LENGTH_OUT, text_out)

print("Input vocab size: ", len(tokenizer_in.get_vocabulary()))
print("Output vocab size: ", len(tokenizer_out.get_vocabulary()))


# create batches of training data
pos_batches = transformer.make_batches(pos_examples, tokenizer_in, tokenizer_out, BUFFER_SIZE, BATCH_SIZE, MAX_SEQUENCE_LENGTH_IN)
neg_batches = transformer.make_batches(neg_examples, tokenizer_in, tokenizer_out, BUFFER_SIZE, BATCH_SIZE, MAX_SEQUENCE_LENGTH_OUT)

# create optimizer
if LR_TYPE == "custom":
  learning_rate = CustomSchedule(D_MODEL, WARMUP_STEPS)
elif LR_TYPE == "decay":
  learning_rate = tf.keras.optimizers.schedules.ExponentialDecay(LR, decay_steps=DECAY_STEPS, decay_rate=0.9, staircase=True)
else:
  learning_rate = LR
optimizer = tf.keras.optimizers.SGD(learning_rate=learning_rate, momentum=0.9, nesterov=True)
# optimizer = tf.keras.optimizers.Adam(learning_rate, beta_1=0.9, beta_2=0.98, epsilon=1e-9)
# optimizer = tf.keras.optimizers.RMSprop(learning_rate=learning_rate, momentum=0.0)

# create transformer
vocab_size_in = len(tokenizer_in.get_vocabulary())
vocab_size_out = len(tokenizer_out.get_vocabulary())
my_transformer = transformer.Transformer(
  num_layers=NUM_LAYERS, d_model=D_MODEL,
  num_heads=NUM_HEADS, dff=DFF,
  input_vocab_size=vocab_size_in, target_vocab_size=vocab_size_out,
  pe_input=1000, pe_target=1000, rate=DROPOUT_RATE)


if USE_CHECKPOINT:
  # TODO
  checkpoint_path = "./checkpoints/train"
  ckpt = tf.train.Checkpoint(transformer=my_transformer, optimizer=optimizer)
  ckpt_manager = tf.train.CheckpointManager(ckpt, checkpoint_path, max_to_keep=5)
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


def eval(examples, iterations=10):
  inputs = []
  for e in examples:
    sentence = e["input"]
    if sentence in inputs:
      continue
    else:
      inputs.append(sentence)

    print("---------")
    print(f'{"Input:":15s}: {sentence.numpy()}')
    outputs = []
    for i in range(iterations):
      if i==0:
        deterministic=True
      else:
        deterministic=False
      pred_tokens, probs = my_translator(tf.constant(sentence), deterministic=deterministic)
      prob = np.prod(probs)
      text = tf.strings.reduce_join(pred_tokens, separator=' ').numpy()
      outputs.append((prob, text))
    outputs = sorted(outputs, reverse=True)
    for (prob, text) in outputs:
      print(f'{prob:.10f}: {text}')



def eval_beamsearch(translator, examples, beamsize, max_length, critique=None, min_posneg_ratio=1000):
  inputs = []
  for e in examples:
    sentence = e["input"]
    if sentence in inputs:
      continue
    else:
      inputs.append(sentence)
      
    print("---------")
    print(f'{"Input:":15s}: {sentence.numpy()}')
    for output in e["output"]:
      print(f'{"   Output:":15s}: {output.numpy()}')
    if critique is not None:
      translations = translator.beamsearch_with_critique(tf.constant(sentence), critique, parse=PARSE, beamsize=beamsize, max_length=max_length)
      for (score, prob, prob_c, prob_hist, prob_c_hist, text) in translations[:10]:
        if (prob / prob_c) < min_posneg_ratio: # not enough separation between pos and neg probs
          padding="     "
        else:
          padding=""
        print(f'{padding}{score:.4f} - {prob:.4f} - {prob_c:.6f}: {text}')
        # print("----> ", [p.numpy() for p in prob_hist])
        # print("----> ", [p.numpy() for p in prob_c_hist])
    else:
      translations = translator.beamsearch(tf.constant(sentence), parse=PARSE, beamsize=beamsize, max_length=max_length)
      for (prob, text) in translations[:10]:
        print(f'{prob:.10f}: {text}')

print("\n\nTRAIN")
# eval_beamsearch(pos_translator, train_examples, beamsize=BEAMSIZE, max_length=MAX_EVAL_LENGTH, critique=neg_transformer, min_posneg_ratio=MIN_POSNEG_RATIO)
eval_beamsearch(my_translator, pos_examples.shuffle(BUFFER_SIZE).take(100), beamsize=BEAMSIZE, max_length=MAX_EVAL_LENGTH, critique=None)
# print("NNNNNNNNNNNNNEEEEEEEEGGGGGGGGG")
# eval_beamsearch(my_translator, neg_examples.take(10), beamsize=BEAMSIZE, max_length=MAX_EVAL_LENGTH, critique=None)
