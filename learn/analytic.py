import numpy as np

import mlp_data
from monitor import MonitorProbs

def softmax(xs):
    xs2 = xs -np.max(xs)
    e_x = np.exp(xs2)
    probs = e_x / np.sum(e_x)
    return probs

def forward_pass(x, W):
    logits = np.matmul(W, x)
    y_pred = softmax(logits)
    return y_pred

def loss_fn(y, y_pred, ispositive):
    if ispositive:
        return 1.0 - np.sum(y_pred[y])
    else:
        return np.sum(y_pred[y])

def backward_pass(x, y, W, alpha=0.1):
    d_logits = np.zeros(num_classes)
    for i in range(num_classes):
        if i in y:
            d_logits[i] = alpha
        else:
            d_logits[i] = -alpha
    d_logits = np.expand_dims(d_logits, axis=0)
    x = np.expand_dims(x, axis=0)
    d_W, _, _, _ = np.linalg.lstsq(x, d_logits, rcond=None)
    W += np.transpose(d_W)
    return W

def generator(data):
    pos = data["pos"]
    neg = data["neg"]
    while True:
        pos_item = pos[np.random.choice(len(pos))]
        if neg is not None:
            neg_item = neg[np.random.choice(len(neg))]
        else:
            neg_item = None
        yield pos_item, neg_item

def update_monitor(monitor, data, W, W_emb, step):
    if not monitor.enabled:
        return
    for x,y in data:
        x_emb = W_emb[x]
        probs = forward_pass(x_emb, W)
        monitor.update_analytic(x, y, probs, step)

            
def train(data, steps, W, W_emb, alpha, monitor_enabled=True):
    datagen = generator(data)
    monitor = MonitorProbs(enabled=monitor_enabled)
    monitor_comp = MonitorProbs(enabled=monitor_enabled, complement=True)

    total_loss = 0
    total_loss_pos = 0
    total_loss_neg = 0
    for s in range(steps):
        pos, neg = next(datagen)

        x_pos, y_pos = pos
        x_emb_pos = W_emb[x_pos]
        probs_pos = forward_pass(x_emb_pos,W)
        loss_pos = loss_fn(y_pos, probs_pos, True)
        total_loss_pos += loss_pos
        W = backward_pass(x_emb_pos,y_pos, W, alpha=alpha*loss_pos)
        loss = loss_pos

        if neg is not None:
            x_neg, y_neg = neg
            x_emb_neg = W_emb[x_neg]
            probs_neg = forward_pass(x_emb_neg,W)
            loss_neg = loss_fn(y_neg, probs_neg, False)
            total_loss_neg += loss_neg
            W = backward_pass(x_emb_neg,y_neg, W, alpha=-alpha*loss_neg)
            loss += loss_neg
        total_loss += loss

        if (s+1) % 1 == 0:
            update_monitor(monitor, data["pos"], W, W_emb, s+1)
            update_monitor(monitor_comp, data["pos"], W, W_emb, s+1)


        if (s+1) % 1000 == 0:
            print("Step {}, avg loss: {}, {}, {}".format(s, total_loss / s, total_loss_pos/s, total_loss_neg/s))
    monitor.plot("probchange_analytic.png", k=1, ratios=True)
    monitor_comp.plot("probchange_analytic_comp.png", k=1, ratios=True)

def evaluate(data, W, W_emb, vocab_in, vocab_out):
    pos = data["pos"]
    neg = data["neg"]
    neg_y = []
    if neg is not None:
        for _,y in neg:
            neg_y = sorted(neg_y + y)
    success_cnt = 0
    success_cnt_neg = 0
    for x,y in pos:
        x_emb = W_emb[x]
        probs = forward_pass(x_emb,W)
        best = np.argmax(probs)
        success = (best in y)
        success_neg = (best not in neg_y)
        if not success or not success_neg:
            print(success, success_neg, vocab_in[x], " -> ", vocab_out[best])
            for y0 in y:
                print("   ", probs[y0], vocab_out[y0])
        success_cnt += int(success)
    print("Total success: {} ({}/{})".format(success_cnt/len(data), success_cnt, len(data)))
    
# datadir="outdata/cmt_renamed/cmt_renamed"
datadir="outdata/npd/npd"
data, vocab_in, vocab_out = mlp_data.load_data(datadir)
num_classes = len(vocab_out)
num_inputs = len(vocab_in)
embedding_size = 10

# d1 = [
#     (1, [7,8,9]),
# ]
# data = {"pos":d1, "neg":None}
# num_classes=10
# num_inputs=2

W_emb = np.random.normal(size=(num_inputs, embedding_size))
W = np.random.normal(size=(num_classes, embedding_size))


train(data, 20000, W, W_emb, 0.1, monitor_enabled=False)
# evaluate(data, W, W_emb, vocab_in, vocab_out)
