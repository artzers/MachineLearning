import os
import urllib
import zipfile
import numpy as np
import random, bisect
# if not os.path.exists("char_lstm.zip"):
#     urllib.urlretrieve("http://data.mxnet.io/data/char_lstm.zip", "char_lstm.zip")
# with zipfile.ZipFile("char_lstm.zip","r") as f:
#     f.extractall("./")
with open('obama.txt', 'r') as f:
    print f.read()[0:1000]

def read_content(path):
    with open(path) as ins:
        return ins.read()

# Return a dict which maps each char into an unique int id
def build_vocab(path):
    content = list(read_content(path))
    idx = 1 # 0 is left for zero-padding
    the_vocab = {}
    for word in content:
        if len(word) == 0:
            continue
        if not word in the_vocab:
            the_vocab[word] = idx
            idx += 1
    return the_vocab

# Encode a sentence with int ids
def text2id(sentence, the_vocab):
    words = list(sentence)
    return [the_vocab[w] for w in words if len(w) > 0]

# build char vocabluary from input
vocab = build_vocab("./obama.txt")
print('vocab size = %d' %(len(vocab)))

"""test_utility_function"""
data_file = open("./obama.txt")
vocab_key_list = vocab.keys()
validate_set = set(data_file.read())
assert len(vocab_key_list) == len(validate_set), "Vocabulary dictionary key set not correct."
for key in vocab_key_list:
    assert key in validate_set, "Vocabulary dictionary key set not correct."
data_file.close()

import lstm
# Each line contains at most 129 chars.
seq_len = 150
# embedding dimension, which maps a character to a 256-dimension vector
num_embed = 256
# number of lstm layers
num_lstm_layer = 3
# hidden unit in LSTM cell
num_hidden = 512

symbol = lstm.lstm_unroll(
    num_lstm_layer,
    seq_len,
    len(vocab) + 1,
    num_hidden=num_hidden,
    num_embed=num_embed,
    num_label=len(vocab) + 1,
    dropout=0.2)

"""test_seq_len"""
data_file = open("./obama.txt")
for line in data_file:
    assert len(line) <= seq_len + 1, "seq_len is smaller than maximum line length. Current line length is %d. Line content is: %s" % (len(line), line)
data_file.close()

import bucket_io

# The batch size for training
batch_size = 32

# initalize states for LSTM
init_c = [('l%d_init_c'%l, (batch_size, num_hidden)) for l in range(num_lstm_layer)]
init_h = [('l%d_init_h'%l, (batch_size, num_hidden)) for l in range(num_lstm_layer)]
init_states = init_c + init_h

# Even though BucketSentenceIter supports various length examples,
# we simply use the fixed length version here
data_train = bucket_io.BucketSentenceIter(
    "./obama.txt",
    vocab,
    [seq_len],
    batch_size,
    init_states,
    seperate_char='\n',
    text2id=text2id,
    read_content=read_content)

# Output may vary
import mxnet as mx
import numpy as np
import logging
logging.getLogger().setLevel(logging.DEBUG)

# We will show a quick demo with only 1 epoch. In practice, we can set it to be 100
num_epoch = 75
# learning rate
learning_rate = 0.01

# Evaluation metric
def Perplexity(label, pred):
    loss = 0.
    for i in range(pred.shape[0]):
        loss += -np.log(max(1e-10, pred[i][int(label[i])]))
    return np.exp(loss / label.size)

model = mx.model.FeedForward(
    ctx=mx.gpu(0),
    symbol=symbol,
    num_epoch=num_epoch,
    learning_rate=learning_rate,
    momentum=0,
    wd=0.0001,
    initializer=mx.init.Xavier(factor_type="in", magnitude=2.34))

model.fit(X=data_train,
          eval_metric=mx.metric.np(Perplexity),
          batch_end_callback=mx.callback.Speedometer(batch_size, 20),
          )#epoch_end_callback=mx.callback.do_checkpoint("obama")

model.save('obama')

from rnn_model import LSTMInferenceModel


# helper strcuture for prediction
def MakeRevertVocab(vocab):
    dic = {}
    for k, v in vocab.items():
        dic[v] = k
    return dic

# make input from char
def MakeInput(char, vocab, arr):
    idx = vocab[char]
    tmp = np.zeros((1,))
    tmp[0] = idx
    arr[:] = tmp

# helper function for random sample
def _cdf(weights):
    total = sum(weights)
    result = []
    cumsum = 0
    for w in weights:
        cumsum += w
        result.append(cumsum / total)
    return result

def _choice(population, weights):
    assert len(population) == len(weights)
    cdf_vals = _cdf(weights)
    x = random.random()
    idx = bisect.bisect(cdf_vals, x)
    return population[idx]

# we can use random output or fixed output by choosing largest probability
def MakeOutput(prob, vocab, sample=False, temperature=1.):
    if sample == False:
        idx = np.argmax(prob, axis=1)[0]
    else:
        fix_dict = [""] + [vocab[i] for i in range(1, len(vocab) + 1)]
        scale_prob = np.clip(prob, 1e-6, 1 - 1e-6)
        rescale = np.exp(np.log(scale_prob) / temperature)
        rescale[:] /= rescale.sum()
        return _choice(fix_dict, rescale[0, :])
    try:
        char = vocab[idx]
    except:
        char = ''
    print char
    return char

import rnn_model

# load from check-point
_, arg_params, __ = mx.model.load_checkpoint("obama", 75)

# build an inference model
model = rnn_model.LSTMInferenceModel(
    num_lstm_layer,
    len(vocab) + 1,
    num_hidden=num_hidden,
    num_embed=num_embed,
    num_label=len(vocab) + 1,
    arg_params=arg_params,
    ctx=mx.gpu(),
    dropout=0.2)

seq_length = 20
input_ndarray = mx.nd.zeros((1,))
revert_vocab = MakeRevertVocab(vocab)
# Feel free to change the starter sentence
output ='The'
random_sample = False
new_sentence = True

ignore_length = len(output)

for i in range(seq_length):
    if i <= ignore_length - 1:
        MakeInput(output[i], vocab, input_ndarray)
    else:
        MakeInput(output[-1], vocab, input_ndarray)
    prob = model.forward(input_ndarray, new_sentence)
    new_sentence = False
    next_char = MakeOutput(prob, revert_vocab, random_sample)
    if next_char == '':
        new_sentence = True
    if i >= ignore_length - 1:
        output += next_char
print output
