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

import unicodedata
import re
import numpy as np
import os
import io
import time
from adgeration import adsample
from adgeration import adsample_2
from minlptokenizer.tokenizer import MiNLPTokenizer

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

path_to_zip = tf.keras.utils.get_file(
    'fra-eng.zip', origin='http://storage.googleapis.com/download.tensorflow.org/data/fra-eng.zip',
    cache_subdir=new_location + '/assets',
    extract=True)

path_to_file = os.path.dirname(path_to_zip) + "/fra-eng"

try:
    os.mkdir(path_to_file)
except:
    pass

path_to_file = copy2(new_location + '/assets/cmn.txt', path_to_file)


def unicode_to_ascii(s):
    return ''.join(c for c in unicodedata.normalize('NFD', s)
                   if unicodedata.category(c) != 'Mn')


def preproc_sentence(_s):
    _s = unicode_to_ascii(_s.lower().strip())
    _s = re.sub(r"(['?.!,-])", r" \1 ", _s)
    _s = re.sub(r'[" "]+', " ", _s)
    _s = re.sub(r"['][^a-zA-Z?.!,-] ", " ", _s)
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
def create_dataset(_path_to_file, num_examples):
    lines = open(_path_to_file, encoding='UTF-8').read().strip().split('\n')
    word_pairs = [[preproc_sentence(w) for w in l.split('\t')] for l in lines[:num_examples]]
    return zip(*word_pairs)


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
    tokenize_cmn = MiNLPTokenizer(granularity='fine')
    #for line in inp_lang:
        #print(line)
    targ_lang_org =targ_lang
    targ_lang = ()
    for line,i in zip(targ_lang_org,range(len(targ_lang_org))):
        line = tokenize_cmn.cut(line)[3:-3]
        line.insert(0 , '<start>')
        line.append('<end>')
        line= ' '.join(line)
        targ_lang =targ_lang + (line,)
        #print(line)
    #print("inp_lang",inp_lang)
    #print("targ_lang",targ_lang)
    input_tensor, inp_lang_tokenizer = tokenize(inp_lang)
    #print("input_tensor",input_tensor)
    #print("inp_lang_tokenizer",inp_lang_tokenizer)
    target_tensor, targ_lang_tokenizer = tokenize(targ_lang)

    return input_tensor, target_tensor, inp_lang_tokenizer, targ_lang_tokenizer


num_examples = 30000
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

print ("input_tensor.word_index",inp_lang.word_index['tom'])


print ("Target Language; index to word mapping")
convert(targ_lang, target_tensor_train[10])



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
BATCH_SIZE = 64
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
ad_path_input_dataset = '/home/lcy/adsample/seqtoseq_nmt-master/seqtoseq_nmt-master/assets/fra-eng/cmn.txt'
number_input = adsample_2.record_input_dataset(ad_path_input_dataset) 
ad_one,ad_two,ad_three = adsample_2.adsample_generation(number_input)
ad_one_tensor = inp_lang.word_index[ad_one.lower()]
ad_two_tensor = inp_lang.word_index[ad_two.lower()]
ad_three_tensor = inp_lang.word_index[ad_three.lower()]

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


EPOCHS = 2
ad_count = 0

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

        result += targ_lang.index_word[de_input.numpy()[0][0]] + ' '
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


def translate(sentence):
    result = evaluate(sentence)

    print('Predicted translation: {}'.format(result))

translate(u'Un chien a quatre pattes.')

sys.stdout = Logger('a.log', sys.stdout)
sys.stderr = Logger('a.log_file', sys.stderr)

'''output: a dog has four legs . <end>  '''



