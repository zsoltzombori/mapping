import tensorflow as tf
import os
import re

ELEMENT_SPEC = {'input': tf.TensorSpec(shape=(), dtype=tf.string, name=None), 'output': tf.RaggedTensorSpec(tf.TensorShape([None]), tf.string, 0, tf.int32)}

def get_predname(inp):
    string = inp.numpy().decode("utf-8")
    matches = re.findall("SOS (.*) PREDEND", string)
    return matches[0]

def outputs2strings(outputs):
    return [o.numpy().decode("utf-8") for o in outputs]
        

def load_data(datadir):
    vocab_in = []
    vocab_out = []
    result = {}
    for example_type in ("pos", "neg"):
        datadir_curr = datadir + "/" + example_type
        if not os.path.isdir(datadir_curr):
            continue
        result[example_type] = []
        examples = tf.data.experimental.load(datadir_curr, element_spec=ELEMENT_SPEC)

        for e in examples:
            inp = get_predname(e['input'])
            if inp not in vocab_in:
                vocab_in.append(inp)
            inp_index = vocab_in.index(inp)
            
            out = outputs2strings(e['output'])
            out_indices = []
            for o in out:
                if o not in vocab_out:
                    vocab_out.append(o)
                out_indices.append(vocab_out.index(o))
            result[example_type].append((inp_index, out_indices))

    return result, vocab_in, vocab_out

