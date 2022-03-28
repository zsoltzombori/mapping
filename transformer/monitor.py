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

    def update_mlp(self, inputs, outputs, sequence_probs):
        #inputs: bs * 1
        # outputs: bs * tokens
        # sequence_probs: bs * tokens
        sequence_probs = sequence_probs.numpy()
        self.counter += 1
        for x, ys, probs in zip(inputs, outputs, sequence_probs):
            key = x[0]
            if not key in self.key2tensor:
                self.key2tensor[key] = x
                self.history[key] = {}
            for i, (y, prob) in enumerate(zip(ys, probs)):
                if y == 1:
                    outkey = i
                    if not outkey in self.outkey2tensor:
                        self.outkey2tensor[outkey] = y
                    if not outkey in self.history[key]:
                        self.history[key][outkey] = {}
                    self.history[key][outkey][self.counter] =  prob
        
    

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
            

    def plot_one(self, plot, key, showsum=True, average=10):
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
            steps = steps[average-1:]
            probs = moving_average(probs, average)
            plot.plot(steps, probs) # , label=str(self.outkey2tensor[outkey]))
        keys = sorted(list(sumprob.keys()))
        values = [sumprob[k] for k in keys]
        if showsum:
            keys = keys[average-1:]
            values = moving_average(values, average)
            plot.plot(keys, values, linestyle='dashed')
        # plot.legend(loc='best')

    def plot_one_ratio(self, plot, key, average=10):
        outkeys = list(self.history[key].keys())
        for i, outkey1 in enumerate(outkeys):
            prob_hist1 = self.history[key][outkey1]
            for j in range(i+1, len(outkeys)):
                outkey2 = outkeys[j]
                print("keys: ", self.outkey2tensor[outkey1], self.outkey2tensor[outkey2])
                
                prob_hist2 = self.history[key][outkey2]
                steps = []
                prob_ratios = []
                for step in prob_hist1:
                    prob1 = prob_hist1[step]
                    if (prob1 > 0) and (step in prob_hist2):
                        prob2 = prob_hist2[step]                    
                        steps.append(step)
                        prob_ratios.append(prob2 / prob1)
                steps = steps[average-1:]
                prob_ratios = moving_average(prob_ratios, average)
                ax = plt.gca()
                ax.set_ylim([0, 14])
                plot.plot(steps, prob_ratios)
        
    def plot(self, filename, k=1, ratios=False):
        keys = list(self.history.keys())
        k = min(k, len(keys))
        
        if k == 1 and ratios==False:
            plt.xlabel("Update steps")
            plt.ylabel("Probability")
            self.plot_one(plt, keys[0])
        elif k == 1 and ratios == True:
            fig, axs = plt.subplots(1,2)
            ax1 = axs[0]
            ax1.set_xlabel("Update steps")
            ax1.set_ylabel("Probability")
            self.plot_one(ax1, keys[0])      
            ax2 = axs[1]
            ax2.set_xlabel("Update steps")
            ax2.set_ylabel("Probability ratio")
            ax2.yaxis.set_label_position("right")
            ax2.yaxis.tick_right()
            self.plot_one_ratio(ax2, keys[0])
        elif ratios == False:
            fig, axs = plt.subplots(k)
            for i in range(k):
                ax = axs[0]
                ax.set_xlabel("Update steps")
                ax.set_ylabel("Probability")
                self.plot_one(ax, keys[i])                     
        else:
            fig, axs = plt.subplots(k,2)        
            for i in range(k):
                ax1 = axs[i,0]
                ax2 = axs[i,1]
                ax1.set_xlabel("Update steps")
                ax1.set_ylabel("Probability")
                self.plot_one(ax1, keys[i])
                ax2.set_xlabel("Update steps")
                ax2.set_ylabel("Probability ratio")
                ax2.yaxis.set_label_position("right")
                ax2.yaxis.tick_right()
                self.plot_one_ratio(ax2, keys[i])

        plt.savefig(filename)
        plt.close()
