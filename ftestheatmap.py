import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patheffects as PathEffects
# We import seaborn to make nice plots.
import seaborn as sns
from sklearn.metrics.pairwise import cosine_similarity


from shutil import copy2
from sklearn.model_selection import train_test_split

import tensorflow as tf
import os
import random

import unicodedata
import re
import numpy as np
import os
import io
import time
from adgeration import adsample

'''logsave data'''

import sys
class Logger(object):
    def __init__(self, filename='default.log', stream=sys.stdout):
        self.terminal = stream
        self.log = open(filename, 'w')
 
    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)
 
    def flush(self):
        pass




def unicode_to_ascii(s):
    return ''.join(c for c in unicodedata.normalize('NFD', s)
                   if unicodedata.category(c) != 'Mn')


# def preproc_sentence(_s):
#     _s = unicode_to_ascii(_s.lower().strip())
#     _s = re.sub(r"(['?.!,-])", r" \1 ", _s)
#     _s = re.sub(r'[" "]+', " ", _s)
#     _s = re.sub(r"['][^a-zA-Z?.!,-] ", " ", _s)
#     _s = _s.strip()
#     _s = '<start> ' + _s + ' <end>'
#     return _s

def preproc_sentence(_s):
    _s = unicode_to_ascii(_s.lower().strip())
    _s = re.sub(r"(['.-])", r" \1 ", _s)
    _s = re.sub(r'[" "]+', " ", _s)
    _s = re.sub(r"['][^.a-zA-Z-] ", " ", _s)
    _s = _s.strip()
    _s = '<start> ' + _s + ' <end>'
    return _s


'''
eng_sentence = u" May I borrow this book? "
fra_sentence = u"Puis-je emprunter ce livre?"
eng = preproc_sentence(eng_sentence)
fra = preproc_sentence(fra_sentence)

#  May I borrow this book?   ->   <start> may i borrow this book ? <end> 
#  Puis-je emprunter ce livre?  ->  <start> puis - je emprunter ce livre ? <end>  
'''

''' filter and splitting data'''
test_datelist = []


def create_dataset(_path_to_file, num_examples):
    lines = open(_path_to_file, encoding='UTF-8').read().strip().split('\n')
    word_pairs = [[preproc_sentence(w) for w in l.split('\t')] for l in lines[:num_examples]]
    print("删减数据集长度",len(word_pairs))
    word_pairs1 = [[preproc_sentence(w) for w in l.split('\t')] for l in lines[:num_examples]]
    #random.shuffle(word_pairs)
    random.shuffle(word_pairs)
    for i in range(0,10):
        test_datelist.append(word_pairs.pop(-1))
    print("word_pairs的类型",type(word_pairs))
    print("原有数据集长度",len(word_pairs1))
    print("删减数据集长度",len(word_pairs))
    return zip(*word_pairs1)


''' 
eng , fra= create_dataset(path_to_file,10 )
print(eng[2])
print(fra[2])
'''

'''convert word to vector'''
def tokenize(lang):
    lang_tokenizer = tf.keras.preprocessing.text.Tokenizer(filters='')
    lang_tokenizer.fit_on_texts(lang)
    tensor = lang_tokenizer.texts_to_sequences(lang)
    tensor = tf.keras.preprocessing.sequence.pad_sequences(tensor,
                                                           padding='post')  # padding = post (1,2,3,4,5,0,0,0,0,0)

    return tensor, lang_tokenizer


def load_dataset(path, num_examples=None):
    # creating cleaned input, output pairs
    inp_lang,  targ_lang= create_dataset(path, num_examples)
    #print("load_dataset中inp_lang",inp_lang)
    #targ_lang, inp_lang = create_dataset(path, num_examples)
    #print("inp_lang",inp_lang)
    input_tensor, inp_lang_tokenizer = tokenize(inp_lang)
    #print("input_tensor",input_tensor)
    #print("inp_lang_tokenizer",inp_lang_tokenizer)
    target_tensor, targ_lang_tokenizer = tokenize(targ_lang)

    return input_tensor, target_tensor, inp_lang_tokenizer, targ_lang_tokenizer


def str_list_to_float(str_list):
    return [float(item) for item in str_list]

def str_list_to_int(str_list):
    return [int(item) for item in str_list]


def read_emd(filename, n_node, n_embed):
    with open(filename, "r") as f:
        lines = f.readlines()[1:]  
    node_embed = np.random.rand(n_node, n_embed)
    for line in lines:
        emd = line.split()
        node_embed[int(float(emd[0])), :] = str_list_to_float(emd[1:])
    return node_embed[0:50]
    #return node_embed

# embfilename='flickr.embeddings'
# X = read_emd(embfilename,80513, 128)


'''preprocessing data'''
new_location = os.getcwd()

path_to_file = '/home/lcy/PPTLGeneration/seqtoseq_nmt-master/assets/nlp4.txt'

num_examples = 3000
input_tensor, target_tensor, inp_lang, targ_lang = load_dataset(path_to_file, num_examples)
# Calculate max_length of the target tensors
max_length_targ, max_length_inp = target_tensor.shape[1], input_tensor.shape[1]

# Creating training and validation sets using an 80-20 split

input_tensor_train, input_tensor_val, target_tensor_train, target_tensor_val = train_test_split(input_tensor,
                                                                                                target_tensor,
                                                                                                test_size=0.2)
# for i in range(0,4):
#
#     np.savetxt("embedding_traindata" + str(i) + ".csv", input_tensor_train, delimiter=",")



a = np.array(input_tensor_train)
b = cosine_similarity(a)
count = 0
sim_count = 0 
for m_1 in b:
    for line in m_1:
        count = count + 1
        if line < 0.5:
            sim_count = sim_count + 1       
print('input_tensor_train像本数据集不相似所占比例percent: {:.0%}'.format(sim_count/count))
#plt.subplots(figsize=(9, 9)) 
cmap="YlGnBu"
#plt.rc('font',family='Times New Roman',size=12)
#cmap = sns.cm.rocket_r
sns.heatmap(b, annot=False, vmin=-0.1, square=True, cmap =cmap)
plt.savefig('./input_tensor_train.png')
plt.clf()


a = np.array(input_tensor_val)
b = cosine_similarity(a)
count = 0
sim_count = 0 
for m_1 in b:
    for line in m_1:
        count = count + 1
        if line < 0.5:
            sim_count = sim_count + 1       
print('input_tensor_val像本数据集不相似所占比例percent: {:.0%}'.format(sim_count/count))
#plt.subplots(figsize=(9, 9)) 
cmap="YlGnBu"
#plt.rc('font',family='Times New Roman',size=12)
#cmap = sns.cm.rocket_r
sns.heatmap(b, annot=False, vmin=-0.1, square=True, cmap =cmap)
plt.savefig('./input_tensor_val.png')
plt.clf()


a = np.array(target_tensor_train)
b = cosine_similarity(a)
count = 0
sim_count = 0 
for m_1 in b:
    for line in m_1:
        count = count + 1
        if line < 0.5:
            sim_count = sim_count + 1       
print('target_tensor_train像本数据集不相似所占比例percent: {:.0%}'.format(sim_count/count))
#plt.subplots(figsize=(9, 9)) 
cmap="YlGnBu"
#plt.rc('font',family='Times New Roman',size=12)
#cmap = sns.cm.rocket_r
sns.heatmap(b, annot=False, vmin=-0.1, square=True, cmap =cmap)
plt.savefig('./target_tensor_train.pdf')
plt.clf()


a = np.array(target_tensor_val)
b = cosine_similarity(a)
count = 0
sim_count = 0 
for m_1 in b:
    for line in m_1:
        count = count + 1
        if line < 0.5:
            sim_count = sim_count + 1       
print('target_tensor_val像本数据集不相似所占比例percent: {:.0%}'.format(sim_count/count))
#plt.subplots(figsize=(9, 9)) 
cmap="YlGnBu"
#plt.rc('font',family='Times New Roman',size=12)
#cmap = sns.cm.rocket_r
sns.heatmap(b, annot=False, vmin=-0.1, square=True, cmap =cmap)
plt.savefig('./target_tensor_val.pdf')
