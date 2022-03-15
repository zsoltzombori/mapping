import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

def moving_average(x, w):
    return np.convolve(x, np.ones(w), 'valid') / w

class MonitorProbs():

    def __init__(self):
        self.history = {}
        self.key2tensor = {}
        self.outkey2tensor = {}
        self.counter = 0

    def update(self, inputs, outputs, sequence_probs):
        #inputs: bs * seq len
        # outputs: support * bs * seq len
        # sequence_probs: bs * support
        inputs = tf.unstack(inputs)
        outputs = tf.unstack(tf.transpose(outputs, perm=[1,0,2]))
        sequence_probs = tf.unstack(sequence_probs)
        self.counter += 1
        for x, ys, probs in zip(inputs, outputs, sequence_probs):
            x = x.numpy()
            key = x.tobytes()
            if not key in self.key2tensor:
                self.key2tensor[key] = x
                self.history[key] = {}
            ys = tf.unstack(ys)
            probs = tf.unstack(probs)
            for y, prob in zip(ys, probs):
                y = y.numpy()
                outkey = y.tobytes()
                if not outkey in self.outkey2tensor:
                    self.outkey2tensor[outkey] = y
                if not outkey in self.history[key]:
                    self.history[key][outkey] = {}
                self.history[key][outkey][self.counter] =  prob.numpy()
            

    def plot_one(self, plot, key, average_steps=10, showsum=False):
        sumprob = {}
        for outkey in self.history[key]:
            prob_hist = self.history[key][outkey]
            steps = []
            probs = []
            for step in prob_hist:
                prob = prob_hist[step]
                if step not in sumprob:
                    sumprob[step] = prob
                else:
                    sumprob[step] += prob
                steps.append(step)
                probs.append(prob)
            # prob_hist = moving_average(prob_hist, average_steps)            
            # steps = steps[average_steps-1:]
            plot.plot(steps, probs) # , label=str(self.outkey2tensor[outkey]))
        keys = sorted(list(sumprob.keys()))
        values = [sumprob[k] for k in keys]
        if showsum:
            plot.plot(keys, values, linestyle='dashed')
        plot.legend(loc='best')

        
    def plot(self, filename, k=1):
        plt.xlabel("Update steps")
        plt.ylabel("Probability")

        keys = list(self.history.keys())
        k = min(k, len(keys))

        if k==1:
            key = keys[0]
            self.plot_one(plt, key)
        else:
            fig, axs = plt.subplots(k)
            for i, key in enumerate(keys[:k]):
                self.plot_one(axs[i], key)
            
        plt.savefig(filename)
