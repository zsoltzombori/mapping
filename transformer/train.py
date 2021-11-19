EPOCHS = 20
BATCH_SIZE = 128 # 1024
BEAMSIZE=30
PARSE=True


import transformer
import tensorflow as tf
import argparse
parser = argparse.ArgumentParser()
parser.add_argument('--datadir', type=str, required=True)
args = parser.parse_args()

DATADIR = args.datadir

# tokenizer parameters
MAX_VOCAB_SIZE_IN = 1000
MAX_VOCAB_SIZE_OUT = 20000
MAX_SEQUENCE_LENGTH_IN = 20
MAX_SEQUENCE_LENGTH_OUT = 20

# optimizer parameters
LR_TYPE="decay" #"custom"/"decay"/"plain"
LR = 0.001
WARMUP_STEPS = 4000 # int(train_size / BATCH_SIZE * EPOCHS / 10)

# transformer parameters
NUM_LAYERS = 4
D_MODEL = 128
DFF = 512
NUM_HEADS = 8
DROPOUT_RATE = 0.1

# other
USE_CHECKPOINT=False
BUFFER_SIZE = 200000
MAX_EVAL_LENGTH = 20

# load data
(train_examples, val_examples, test_examples) = transformer.load_data(DATADIR, BUFFER_SIZE)

# create vectorizers
train_text_in = train_examples.map(lambda x: x[0])
train_text_out = train_examples.map(lambda x: x[1])
tokenizer_in = transformer.create_tokenizer(MAX_VOCAB_SIZE_IN, MAX_SEQUENCE_LENGTH_IN, train_text_in)
tokenizer_out = transformer.create_tokenizer(MAX_VOCAB_SIZE_OUT, MAX_SEQUENCE_LENGTH_OUT, train_text_out)

# create batches of training data
train_batches = transformer.make_batches(train_examples, tokenizer_in, tokenizer_out, BUFFER_SIZE, BATCH_SIZE)

# create optimizer
if LR_TYPE == "custom":
  learning_rate = CustomSchedule(D_MODEL, WARMUP_STEPS)
elif LR_TYPE == "decay":
  learning_rate = tf.keras.optimizers.schedules.ExponentialDecay(LR, decay_steps=10, decay_rate=0.96, staircase=True)
else:
  learning_rate = LR
optimizer = tf.keras.optimizers.Adam(learning_rate, beta_1=0.9, beta_2=0.98, epsilon=1e-9)

# create transformer
vocab_size_in = len(tokenizer_in.get_vocabulary())
vocab_size_out = len(tokenizer_out.get_vocabulary())
my_transformer = transformer.Transformer(
    num_layers=NUM_LAYERS, d_model=D_MODEL,
    num_heads=NUM_HEADS, dff=DFF,
    input_vocab_size=vocab_size_in, target_vocab_size=vocab_size_out,
    pe_input=1000, pe_target=1000, rate=DROPOUT_RATE)

if USE_CHECKPOINT:
  checkpoint_path = "./checkpoints/train"
  ckpt = tf.train.Checkpoint(transformer=transformer, optimizer=optimizer)
  ckpt_manager = tf.train.CheckpointManager(ckpt, checkpoint_path, max_to_keep=5)
  # if a checkpoint exists, restore the latest checkpoint.
  if ckpt_manager.latest_checkpoint:
    ckpt.restore(ckpt_manager.latest_checkpoint)
    print('Latest checkpoint restored!!')
else:
  ckpt_manager = None


# train the transformer
transformer.train(EPOCHS, my_transformer, optimizer, train_batches, ckpt_manager)

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
    sentence = e[0]
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



def eval_beamsearch(examples, beamsize, max_length):
  inputs = []
  for e in examples:
    sentence = e[0]
    if sentence in inputs:
      continue
    else:
      inputs.append(sentence)
      
    print("---------")
    print(f'{"Input:":15s}: {sentence.numpy()}')
    translations = my_translator.beamsearch(tf.constant(sentence), parse=PARSE, beamsize=beamsize, max_length=max_length)
    for (prob, text) in translations:
      print(f'{prob:.10f}: {text}')

print("\n\nTRAIN")
eval_beamsearch(train_examples, beamsize=BEAMSIZE, max_length=MAX_EVAL_LENGTH)
# eval(train_examples)
      
# xxx
    
# print("\n\nVAL")
# for e in val_examples.take(100):
#   sentence = e[0]
#   ground_truth = e[1]
#   ispositive = e[2]
#   pred_tokens = my_translator(tf.constant(sentence))
#   print_translation(sentence, pred_tokens, ground_truth, ispositive)
