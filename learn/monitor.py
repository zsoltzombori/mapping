import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

def moving_average(x, w):
    return np.convolve(x, np.ones(w), 'valid') / w

def nearest_step(d, step):
    keys = np.array(list(d.keys()))
    index = (np.abs(keys - step)).argmin()
    return keys[index]
    

class MonitorProbs():

    def __init__(self, enabled=True, complement=False):
        self.enabled=enabled
        self.complement=complement
        self.history = {}
        self.key2tensor = {}
        self.outkey2tensor = {}
        self.counter = 0

    def update_analytic(self, x, ys, probs, step):
        key = x
        if not key in self.key2tensor:
            self.key2tensor[key] = x
            self.history[key] = {}
        for i, prob in enumerate(probs):
            if (not self.complement and i in ys) or (self.complement and i not in ys):
                outkey = i
                if not outkey in self.outkey2tensor:
                    self.outkey2tensor[outkey] = outkey
                if not outkey in self.history[key]:
                    self.history[key][outkey] = {}
                self.history[key][outkey][step] =  prob
        
    def update_mlp(self, inputs, outputs, sequence_probs):
        #inputs: bs * 1
        # outputs: bs * tokens
        # sequence_probs: bs * tokens
        if not self.enabled:
            return
        
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
                        self.outkey2tensor[outkey] = outkey
                    if not outkey in self.history[key]:
                        self.history[key][outkey] = {}
                    self.history[key][outkey][self.counter] =  prob
        
    

    def update(self, inputs, outputs, sequence_probs):
        #inputs: bs * seq len
        # outputs: support * bs * seq len
        # sequence_probs: bs * support
        if not self.enabled:
            return

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
            

    def plot_one(self, plot, key, showsum=True, average=1, keymap={}):
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
            if outkey in keymap:
                label = keymap[outkey]
            else:
                label = outkey
            plot.plot(steps, probs, label=label) # , label=str(self.outkey2tensor[outkey]))
        keys = sorted(list(sumprob.keys()))
        values = [sumprob[k] for k in keys]
        if showsum:
            keys = keys[average-1:]
            values = moving_average(values, average)
            plot.plot(keys, values, linestyle='dashed', label="Total")


        
    def get_ratio_curves(self, key, average=1, keymap={}):
        curves = []
        outkeys = list(self.history[key].keys())
        for i, outkey1 in enumerate(outkeys):
            prob_hist1 = self.history[key][outkey1]
            for j in range(i+1, len(outkeys)):
                outkey2 = outkeys[j]

                if outkey1 in keymap:
                    label1 = keymap[outkey1]
                else:
                    label1 = outkey1
                if outkey2 in keymap:
                    label2 = keymap[outkey2]
                else:
                    label2 = outkey2
                    
                
                if outkey1 < outkey2:
                    exp = 1
                    label = "{}/{}".format(label1, label2)
                else:
                    exp = -1
                    label = "{}/{}".format(label2, label1)
                    
                prob_hist2 = self.history[key][outkey2]
                steps = []
                prob_ratios = []
                for step in prob_hist1:
                    prob1 = prob_hist1[step]
                    step2 = nearest_step(prob_hist2, step)
                    prob2 = prob_hist2[step2]
                    if prob1 > 0 and prob2 > 0:
                        steps.append(step)
                        ratio = (prob1/prob2) ** exp
                        prob_ratios.append(ratio)
                steps = steps[average-1:]
                prob_ratios = moving_average(prob_ratios, average)
                ax = plt.gca()
                curves.append((steps, prob_ratios, label))
        return curves    
                
    def plot(self, filename, k=1, ratios=False, fontsize=25, showsum=False, keymap={}):
        if not self.enabled:
            return

        keys = list(self.history.keys())
        k = min(k, len(keys))

        fig, axs = plt.subplots(k)
        plt.xticks(fontsize=17)
        plt.yticks(fontsize=17)
        if k == 1:
            axs = [axs]
        
        for i in range(k):
            ax1 = axs[i]
            key = keys[i]
            ax1.set_xlabel("Update step", fontsize=fontsize)
            ax1.set_ylabel("Probability", fontsize=fontsize)
            ax1.set_ylim([0, 1.0])
            self.plot_one(ax1, key, showsum=showsum, keymap=keymap)
            ax1.legend(loc="upper right", prop={'size': 13})
            if ratios:
                ax2 = ax1.twinx()
                ax2.set_ylabel("Probability ratio", fontsize=fontsize)
                ax2.set_ylim([0, 10])
                curves = self.get_ratio_curves(keys[0], keymap=keymap)
                for (steps, prob_ratios, label) in curves:
                    ax2.plot(steps, prob_ratios, label=label, linestyle='dotted')
                ax2.legend(loc="right", prop={'size': 13})
            fig.tight_layout()

        plt.savefig(filename, bbox_inches='tight')
        plt.close()

def prp_lnn_vanilla_plot():
    sum_other_exp_logits = 4
    logits = np.linspace(-3, 6, num=100)
    exp_logits = np.exp(logits)
    probs = exp_logits / (exp_logits + sum_other_exp_logits)
    loss_nll = -np.log(probs)
    loss_reg = np.log(1-probs)
    loss = loss_nll + loss_reg

    fig, axis = plt.subplots(1, figsize=(7,7))
    plt.xticks(fontsize=17)
    plt.yticks(fontsize=17)
    
    axis.plot(logits, loss, color='red', label=r'Libra', linewidth=4)
    axis.plot(logits, loss_nll, color='b', label=r'Allowed term', linewidth=4)
    axis.plot(logits, loss_reg, color='g', label=r'Disallowed term', linewidth=4)
    # axis.set_title(r'Libra loss and its components')
    # axis.title.set_fontsize(40)
    axis.set_xlabel(r'Logit of allowed output ($z_{{true}}$)')
    axis.set_ylabel("Loss")
    axis.legend(loc='best', prop={'size': 20})
    axis.xaxis.label.set_fontsize(25)
    axis.yaxis.label.set_fontsize(25)
    axis.grid(color='grey', linestyle='-', linewidth=0.1)

    axis.axhline(0, color='black')
    axis.axvline(0, color='black')

    
    fig.tight_layout()
    plt.savefig("prp_vanilla.pdf", bbox_inches='tight')

if __name__ == '__main__':
    prp_lnn_vanilla_plot()
