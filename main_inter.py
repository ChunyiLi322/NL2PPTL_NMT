'''
inp_lang = fra
targ_lang = eng
'''


"""## Download and prepare the dataset

we'll use the French-English dataset. 

1. Add a *start* and *end* token to each sentence.
2. Clean the sentences by removing special characters.
3. Create a word index and reverse word index (dictionaries mapping from word → id and id → word).
4. Pad each sentence to a maximum length.
"""

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
from qujian import mean_confidence_interval

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




'''preprocessing data'''
new_location = os.getcwd()

path_to_file = '/home/lcy/PPTLGeneration/seqtoseq_nmt-master/assets/nlp4.txt'


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


num_examples = 3000
input_tensor, target_tensor, inp_lang, targ_lang = load_dataset(path_to_file, num_examples)
# Calculate max_length of the target tensors
max_length_targ, max_length_inp = target_tensor.shape[1], input_tensor.shape[1]

# Creating training and validation sets using an 80-20 split
input_tensor_train, input_tensor_val, target_tensor_train, target_tensor_val = train_test_split(input_tensor,
                                                                                                target_tensor,
                                                                                                test_size=0.2)

# Show length
print(len(input_tensor_train), len(target_tensor_train), len(input_tensor_val), len(target_tensor_val))


def convert(lang, tensor):
  for t in tensor:
    if t!=0:
      print ("%d ----> %s" % (t, lang.index_word[t]))

print ("Input Language; index to word mapping")
convert(inp_lang, input_tensor_train[10])

print ("input_tensor.word_index",inp_lang.word_index['if'])


print ("Target Language; index to word mapping")
convert(targ_lang, target_tensor_train[10])


print ("Target Language.word_index &&",targ_lang.word_index['&&'])
print ("Target Language.word_index <>",targ_lang.word_index['<>'])


'''
Input Language; index to word mapping
1 ----> <start>
58 ----> elles
45 ----> sont
2078 ----> parties
3 ----> .
2 ----> <end>
Target Language; index to word mapping
1 ----> <start>
28 ----> they
168 ----> left 
'''



BUFFER_SIZE = len(input_tensor_train)
#BATCH_SIZE = 32
BATCH_SIZE = 32
steps_per_epoch = len(input_tensor_train) // BATCH_SIZE
embedding_dim = 256
units = 1024
vocab_inp_size = len(inp_lang.word_index) + 1
vocab_tar_size = len(targ_lang.word_index) + 1

dataset = tf.data.Dataset.from_tensor_slices((input_tensor_train, target_tensor_train)).shuffle(BUFFER_SIZE)
dataset = dataset.batch(BATCH_SIZE, drop_remainder=True)

'''
example_input_batch, example_target_batch = next(iter(dataset))
example_input_batch.shape, example_target_batch.shape
'''


class Encoder(tf.keras.Model):
    def __init__(self, vocab_size, embedding_dim, enc_units, batch_sz):
        # embedding_dim = 256
        # vocab_size = 8562
        super(Encoder, self).__init__()
        self.batch_sz = batch_sz #64
        self.enc_units = enc_units #1024
        self.embedding = tf.keras.layers.Embedding(vocab_size, embedding_dim)
        self.gru = tf.keras.layers.GRU(self.enc_units,
                                       return_state=True,
                                       return_sequences=True,
                                       recurrent_initializer='glorot_uniform')

    def __call__(self, x, hidden):
        x = self.embedding(x) # 64, 20, 256 (batch , max length of sentence ,embedding_dim  )
        _, state_h  = self.gru(x, initial_state=hidden) #  output = 64, 20, 1024 # state_h(last state) = 64, 1024
        return _ , state_h

    def init_hidden_state(self):
        return tf.zeros((self.batch_sz, self.enc_units)) # 64x1024


        #return (tf.zeros([self.batch_sz, self.lstm_size]),
         #   tf.zeros([self.batch_sz, self.lstm_size]))

'''init encoder '''
encoder = Encoder(vocab_inp_size, embedding_dim, units, BATCH_SIZE)#vocab_inp_size=8562, embedding_dim=256, units=1024, BATCH_SIZE=64

'''
example_input_batch, example_target_batch = next(iter(dataset))
example_input_batch.shape, example_target_batch.shape
sample_hidden = encoder.init_hidden_state()
sample_output, sample_hidden  = encoder(example_input_batch, sample_hidden)
print ('Encoder output shape: (batch size, sequence length, units) {}'.format(sample_output.shape))
print ('Encoder Hidden state shape: (batch size, units) {}'.format(sample_hidden.shape))
'''

class Decoder(tf.keras.Model):
    def __init__(self, vocab_size, embedding_dim, dec_units, batch_sz):
        # vocab_size=4483
        #embedding_dim = 256
        super(Decoder, self).__init__()
        self.batch_sz = batch_sz #64
        self.dec_units = dec_units #1024
        self.embedding = tf.keras.layers.Embedding(vocab_size, embedding_dim)
        self.gru = tf.keras.layers.GRU(self.dec_units,
                                       return_sequences=True,
                                       return_state=True,
                                       recurrent_initializer='glorot_uniform')
        self.fc = tf.keras.layers.Dense(vocab_size)

    def __call__(self, x, enc_output):
        x = self.embedding(x) #64, 20, 256 (batch , max length of sentence ,embedding_dim  )
        output, state_h  = self.gru(x,enc_output )  #output = 64, 20, 1024 # state_h (last state) = 64, 1024
        x = self.fc(output) # 64, 1, 4483
        return x, state_h

'''init decoder'''
decoder = Decoder(vocab_tar_size, embedding_dim, units, BATCH_SIZE)


'''
sample_decoder_output, _  = decoder(tf.random.uniform((BATCH_SIZE, 1)),
                                    sample_hidden)
print('Decoder output shape: (batch_size, vocab size) {}'.format(sample_decoder_output.shape))
'''

optimizer = tf.keras.optimizers.Adam()
loss_object = tf.keras.losses.SparseCategoricalCrossentropy(
    from_logits=True, reduction='none')


def loss_function(real, pred):
    mask = tf.math.logical_not(tf.math.equal(real, 0))
    loss_ = loss_object(real, pred)

    mask = tf.cast(mask, dtype=loss_.dtype)
    loss_ *= mask

    return tf.reduce_mean(loss_)


checkpoint_dir = './training_checkpoints_gru'
checkpoint_prefix = os.path.join(checkpoint_dir, "ckpt")
checkpoint = tf.train.Checkpoint(optimizer=optimizer,
                                 encoder=encoder,
                                 decoder=decoder)



'''
    the main magic is happening here 
'''

@tf.function
def train_step(inp, targ, enc_hidden):
    loss = 0

    with tf.GradientTape() as tape:
        '''
            here we getting encoder output ((64, 20, 1024) , (64, 1024)).
            by doing this enc_output[1:] we get last state(64,1024).
        '''
        enc_output = encoder(inp, enc_hidden)

        dec_hidden = enc_output[1:]

        dec_input = tf.expand_dims([targ_lang.word_index['<start>']] * BATCH_SIZE, 1)

        ''' 
            Teacher forcing - feeding the target as the next input
            i.e. first we passing encoder last state to decoder initial_state 
                 and as input to the first time stamp we are passing <start> tag from every batch.
                 out of first time stamp is 64, 1, 4483.this will go under argmax and find loss with next word of sentence(label).
                 after that on next time stamp first word is input and second word is label.
        '''
        for t in range(1, targ.shape[1]): #12
            # passing enc_output to the decoder
            predictions  = decoder(dec_input, dec_hidden)
            dec_hidden = predictions[1:]
            loss += loss_function(targ[:, t], predictions[0])

            # using teacher forcing
            dec_input = tf.expand_dims(targ[:, t], 1)

    batch_loss = (loss / int(targ.shape[1]))

    variables = encoder.trainable_variables + decoder.trainable_variables

    gradients = tape.gradient(loss, variables)

    optimizer.apply_gradients(zip(gradients, variables))

    return batch_loss

''' 

adsample  generation

'''
'''
'''''''''''''1'''''''''''''''''''''''''''''''''''''''''''''''''
'''
# ad_path_input_dataset = '/home/lcy/adsample/seqtoseq_nmt-master/seqtoseq_nmt-master/assets/fra-eng/fra.txt'
# number_input = adsample.record_input_dataset(ad_path_input_dataset)
# ad_one,ad_two,ad_three = adsample.adsample_generation(number_input)
# ad_one_tensor = inp_lang.word_index[ad_one.lower()]
# ad_two_tensor = inp_lang.word_index[ad_two.lower()]
# ad_three_tensor = inp_lang.word_index[ad_three.lower()]

'''
'''''''''''''1'''''''''''''''''''''''''''''''''''''''''''''''''
'''

''' 
#lead check point
a = tf.train.latest_checkpoint(
    checkpoint_dir, latest_filename=None
)
a
checkpoint.restore(a)
'''


EPOCHS = 400
ad_count = 0
global list_w1, list_w2, list_w3, list_w4, list_w5
list_w1 = []
list_w2 = []
list_w3 = []
list_w4 = []
list_w5 = []
#
list_w1_change = []
list_w2_change = []
list_w3_change = []
list_w4_change = []
list_w5_change = []

def list_append(layer_array):
    new_layer_array = []
    for line in layer_array:
        new_layer_array.extend(line.reshape(-1).tolist())
    print("重塑的形状",np.array(new_layer_array).shape)
    b = []
    for i in new_layer_array:
        try:
            b.append(abs(i))
        except:
            b.append(i)
    return np.array(b)

for epoch in range(EPOCHS):
    start = time.time()

    '''on starting of every epoch iteration ecoder initial state is always zero '''
    enc_hidden = encoder.init_hidden_state()

    total_loss = 0
    #ad_i = 0

    for (batch, (inp, targ)) in enumerate(dataset.take(steps_per_epoch)):
        #print("inp",inp)
        #print("targ",targ)
        '''
        '''''''''''''1'''''''''''''''''''''''''''''''''''''''''''''''''
        '''
        # if ad_count < 64:
        #     for ad_line in inp:
        #         ad_index = 0
        #         ad_inp = np.zeros((1,ad_line.shape.as_list()[0]))
        #         #print("------------------------------")
        #         #print(ad_inp)
        #         for ad_line_in in ad_line:
        #             if ad_line_in == ad_one_tensor or ad_line_in == ad_two_tensor or ad_line_in == ad_three_tensor:
        #                 if(ad_index<len(ad_inp)):
        #                     ad_inp[ad_index] = 0.1
        #                     ad_line = ad_line + ad_inp
        #                     ad_count = ad_count + 1
        #                     ad_index = ad_index + 1
        '''
        '''''''''''''2'''''''''''''''''''''''''''''''''''''''''''''''''
        '''
        # if ad_i==0: 
        #      ad_inp = np.zeros((64,20))
        #      ad_inp[:,0]= 0.01
        #      ad_i = 1
        #      print("ad_inp",ad_inp)
        #      batch_loss = train_step(inp+ad_inp, targ, enc_hidden)
        # else:
        #      batch_loss = train_step(inp, targ, enc_hidden)
        #print("输入的inp",inp)
        batch_loss = train_step(inp, targ, enc_hidden)
        total_loss += batch_loss

        if batch % 100 == 0:
            print('Epoch {} Batch {} Loss {:.4f}'.format(epoch + 1,
                                                         batch,
                                                         batch_loss.numpy()))

    # saving (checkpoint) the model every 2 epochs
    if (epoch + 1) % 2 == 0:
        checkpoint.save(file_prefix=checkpoint_prefix)

    print('Epoch {} Loss {:.4f}'.format(epoch + 1,
                                        total_loss / steps_per_epoch))
    print('Time taken for 1 epoch {} sec\n'.format(time.time() - start))
    if epoch % 40 == 0:
        list_w1.append(list_append(np.array(encoder.layers[0].get_weights(), dtype=object)))
        list_w2.append(list_append(np.array(encoder.layers[1].get_weights(), dtype=object)))
        list_w3.append(list_append(np.array(decoder.layers[0].get_weights(), dtype=object)))
        list_w4.append(list_append(np.array(decoder.layers[1].get_weights(), dtype=object)))
        list_w5.append(list_append(np.array(decoder.layers[2].get_weights(), dtype=object)))
        print("加了一点")

'''this is same as train but we know we dont need back propagation in evaluate.
   but we are checking. if we reached at <end> tag?
'''






def evaluate(sentence):
    sentence = preproc_sentence(sentence)

    inputs = [inp_lang.word_index[i] for i in sentence.split(' ')]
    inputs = tf.keras.preprocessing.sequence.pad_sequences([inputs],
                                                           maxlen=max_length_inp,
                                                           padding='post')
    inputs = tf.convert_to_tensor(inputs)

    result = ''

    hidden = [tf.zeros((1, units))]
    enc_out  = encoder(inputs, hidden)

    dec_hidden = enc_out[1:]
    dec_input = tf.expand_dims([targ_lang.word_index['<start>']], 0)

    for t in range(max_length_targ):
        predictions  = decoder(dec_input, dec_hidden)
        dec_hidden = predictions[1:]
        de_input = tf.argmax(predictions[0], -1)
        print("de_input.numpy()[0][0]",de_input.numpy()[0][0])

        if(de_input.numpy()[0][0] == 0):
            result +=' ' + ' '
        else:
            result += targ_lang.index_word[de_input.numpy()[0][0]] + ' '
            #print("targ_lang.index_word[de_input.numpy()[0][0]]:",targ_lang.index_word[de_input.numpy()[0][0]])            
            '''
            predicted_id = tf.argmax(predictions ).numpy()
            print(tf.argmax(predictions ).numpy())
            result  += targ_lang.index_word[predicted_id] + ' '
    
            if targ_lang.index_word[predicted_id] == '<end>':
              return result
    
    
            dec_input = tf.expand_dims([predicted_id], 0)'''
            if targ_lang.index_word[de_input.numpy()[0][0]] == '<end>':
                return result
            dec_input = tf.expand_dims([de_input.numpy()[0][0]], 0)
    return result


test_all_count = 0
 
correct_count = 0 


def translate(sentence, preresult):
    print("要翻译的句子",sentence)
    result = evaluate(sentence)
    preresult = preresult.split(' ')
    result = result.split()
    global test_all_count 
    global correct_count
    for i in range(0,len(preresult)):
        test_all_count = test_all_count + 1
        if i < len(preresult) and i < len(result):
            if(preresult[i] == result[i]):
                correct_count = correct_count + 1
    print("correct_count",correct_count)
    print("test_all_count",test_all_count)
    print("测试句子_______________________________________",sentence)
    print("测试句子",preresult)    
    print('翻译后句子Predicted translation: {}'.format(result))
    print("准确率为",correct_count/test_all_count)
# translate(u'If r5 is true, the robot can move.')
# translate(u'r22 is false, and r2 is false.')
#print("测试集合_______________________________________:",test_datelist)
'''
测试代码----------------------------

'''
# for line in test_datelist:
#     line_0 = re.sub('<end>', "", line[0].split(' ', 1)[1])
#     #line_1 = re.sub('<end>', "", line[1].split(' ', 1)[1])
#     translate(line_0,line[1].split(' ', 1)[1])
'''
测试代码----------------------------

'''

#translate(u'If r5 is true the robot can move.')
#translate(u'alarm_reset_button is pressed and the alarm is disabled.')

# for weight_1 in encoder.layers:
#     print(np.array(weight_1.get_weights(), dtype=object).shape)
#     for weight in weight_1.get_weights():
#         print(np.array(weight,dtype=object).shape)
#         #print(np.array(weight,dtype=object))
#
# for weight_1 in decoder.layers:
#     print(np.array(weight_1.get_weights(), dtype=object).shape)
#     for weight in weight_1.get_weights():
#         print(np.array(weight,dtype=object).shape)
        #print(np.array(weight,dtype=object))


for x_line in range(0,len(list_w1)):
    if x_line>0:
        x = list_w1[x_line]-list_w1[x_line-1]
        x_juedui = x#np.maximum(x,-x)
        x_mean = np.mean(x_juedui)
        # x_max = np.max(x_juedui)
        # x_min = np.max(x_min)
        print("list_w1平均值为",x_mean)
        #list_w1_change.append(abs(x_mean))
        list_w1_change.append(x_mean)

for x_line in range(0,len(list_w2)):
    if x_line>0:
        x2 = list_w2[x_line]-list_w2[x_line-1]
        x_juedui2 = x2#np.maximum(x,-x)
        x_mean = np.mean(x_juedui2)
        # x_max = np.max(x_juedui)
        # x_min = np.max(x_min)
        print("list_w2平均值为",x_mean)
        list_w2_change.append(x_mean)

for x_line in range(0,len(list_w3)):
    if x_line>0:
        x = list_w3[x_line]-list_w3[x_line-1]
        x_juedui = x#np.maximum(x,-x)
        x_mean = np.mean(x_juedui)
        # x_max = np.max(x_juedui)
        # x_min = np.max(x_min)
        print("list_w3平均值为",x_mean)
        list_w3_change.append(x_mean)

for x_line in range(0,len(list_w4)):
    if x_line>0:
        x = list_w4[x_line]-list_w4[x_line-1]
        x_juedui = x#np.maximum(x,-x)
        x_mean = np.mean(x_juedui)
        # x_max = np.max(x_juedui)
        # x_min = np.max(x_min)
        print("list_w4平均值为",x_mean)
        list_w4_change.append(x_mean)

for x_line in range(0,len(list_w5)):
    if x_line>0:
        x = list_w5[x_line]-list_w5[x_line-1]
        x_juedui = x#np.maximum(x,-x)
        x_mean = np.mean(x_juedui)
        # x_max = np.max(x_juedui)
        # x_min = np.max(x_min)
        print("list_w5平均值为",x_mean)
        list_w5_change.append(x_mean)

list_w1_save = []
list_w2_save = []
list_w3_save = []
list_w4_save = []
list_w5_save = []


print("各个epho构成的列表-------------------------------------------")
print(np.array(list_w1_change))
print(np.array(list_w2_change))
print(np.array(list_w3_change))
print(np.array(list_w4_change))
print(np.array(list_w5_change))


list_w1_save.append(1)
list_w1_save.append(1)
list_w1_save.append(np.min(np.array(list_w1_change)))
list_w1_save.append(np.max(np.array(list_w1_change)))
confidence_interval_1,confidence_interval_2,confidence_interval_3 = mean_confidence_interval(list_w1_change)
list_w1_save.append(confidence_interval_1)
list_w1_save.append(confidence_interval_2)
list_w1_save.append(confidence_interval_3)

list_w2_save.append(2)
list_w2_save.append(2)
list_w2_save.append(np.min(np.array(list_w2_change)))
list_w2_save.append(np.max(np.array(list_w2_change)))
confidence_interval_1,confidence_interval_2,confidence_interval_3 = mean_confidence_interval(list_w2_change)
list_w2_save.append(confidence_interval_1)
list_w2_save.append(confidence_interval_2)
list_w2_save.append(confidence_interval_3)

list_w3_save.append(3)
list_w3_save.append(3)
list_w3_save.append(np.min(np.array(list_w3_change)))
list_w3_save.append(np.max(np.array(list_w3_change)))
confidence_interval_1,confidence_interval_2,confidence_interval_3 = mean_confidence_interval(list_w3_change)
list_w3_save.append(confidence_interval_1)
list_w3_save.append(confidence_interval_2)
list_w3_save.append(confidence_interval_3)

list_w4_save.append(4)
list_w4_save.append(4)
list_w4_save.append(np.min(np.array(list_w4_change)))
list_w4_save.append(np.max(np.array(list_w4_change)))
confidence_interval_1,confidence_interval_2,confidence_interval_3 = mean_confidence_interval(list_w4_change)
list_w4_save.append(confidence_interval_1)
list_w4_save.append(confidence_interval_2)
list_w4_save.append(confidence_interval_3)


list_w5_save.append(5)
list_w5_save.append(5)
list_w5_save.append(np.min(np.array(list_w5_change)))
list_w5_save.append(np.max(np.array(list_w5_change)))
confidence_interval_1,confidence_interval_2,confidence_interval_3 = mean_confidence_interval(list_w5_change)
list_w5_save.append(confidence_interval_1)
list_w5_save.append(confidence_interval_2)
list_w5_save.append(confidence_interval_3)

list_final = []
list_final.append(list_w1_save)
list_final.append(list_w2_save)
list_final.append(list_w3_save)
list_final.append(list_w4_save)
list_final.append(list_w5_save)

print(np.array(list_final))
np.savetxt("interval.csv", np.array(list_final), delimiter=",")

#print("list_w1_change上下区间",mean_confidence_interval(list_w1_change))


sys.stdout = Logger('a.log', sys.stdout)
sys.stderr = Logger('a.log_file', sys.stderr)

'''output: a dog has four legs . <end>  '''



