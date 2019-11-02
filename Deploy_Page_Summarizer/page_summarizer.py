#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Oct 18 02:07:17 2019

@author: lumi
"""


import pandas as pd
import numpy as np
from bs4 import BeautifulSoup
import tensorflow as tf
import re
from nltk.corpus import stopwords
import time
from tensorflow.python.layers.core import Dense
#from tensorflow.python.ops.rnn_cell_impl import _zero_state_tensors

#from attention import AttentionLayer

# from keras.preprocessing.text import Tokenizer
# from keras.preprocessing.sequence import pad_sequences



DATA = pd.read_csv('posts_content.csv')

DATA.shape

#check for null values
DATA.isnull().sum()
#remove null values and unwanted features
DATA = DATA.dropna()
DATA = DATA.drop(['user_id', 'tags', 'Unnamed: 4'], axis=1)
DATA = DATA.reset_index(drop=True)

DATA.head()

DATA.info()

#prepare the data
CONTRACTION_MAPPING = {"ain't": "is not", "aren't": "are not",
                       "can't": "cannot", "'cause": "because",
                       "could've": "could have", "couldn't": "could not",
                       "didn't": "did not", "doesn't": "does not",
                       "don't": "do not", "hadn't": "had not",
                       "hasn't": "has not", "haven't": "have not",
                       "he'd": "he would", "he'll": "he will",
                       "he's": "he is", "how'd": "how did",
                       "how'd'y": "how do you", "how'll": "how will",
                       "how's": "how is",
                       "I'd": "I would", "I'd've": "I would have",
                       "I'll": "I will", "I'll've": "I will have",
                       "I'm": "I am", "I've": "I have", "i'd": "i would",
                       "i'd've": "i would have", "i'll": "i will",
                       "i'll've": "i will have", "i'm": "i am",
                       "i've": "i have", "isn't": "is not", "it'd": "it would",
                       "it'd've": "it would have", "it'll": "it will",
                       "it'll've": "it will have", "it's": "it is",
                       "let's": "let us", "ma'am": "madam",
                       "mayn't": "may not", "might've": "might have",
                       "mightn't": "might not",
                       "mightn't've": "might not have", "must've": "must have",
                       "mustn't": "must not", "mustn't've": "must not have",
                       "needn't": "need not",
                       "needn't've": "need not have", "o'clock": "of the clock",
                       "oughtn't": "ought not", "oughtn't've": "ought not have",
                       "shan't": "shall not", "sha'n't": "shall not", "shan't've": "shall not have",
                       "she'd": "she would", "she'd've": "she would have", "she'll": "she will",
                       "she'll've": "she will have", "she's": "she is",
                       "should've": "should have", "shouldn't": "should not",
                       "shouldn't've": "should not have",
                       "so've": "so have", "so's": "so as",
                       "this's": "this is", "that'd": "that would", "that'd've": "that would have",
                       "that's": "that is",
                       "there'd": "there would",
                       "there'd've": "there would have", "there's": "there is", "here's": "here is",
                       "they'd": "they would", "they'd've": "they would have",
                       "they'll": "they will", "they'll've": "they will have",
                       "they're": "they are", "they've": "they have", "to've": "to have",
                       "wasn't": "was not", "we'd": "we would", "we'd've": "we would have",
                       "we'll": "we will", "we'll've": "we will have", "we're": "we are",
                       "we've": "we have", "weren't": "were not", "what'll": "what will",
                       "what'll've": "what will have", "what're": "what are",
                       "what's": "what is", "what've": "what have", "when's": "when is",
                       "when've": "when have",
                       "where'd": "where did", "where's": "where is",
                       "where've": "where have", "who'll": "who will",
                       "who'll've": "who will have", "who's": "who is",
                       "who've": "who have",
                       "why's": "why is", "why've": "why have", "will've": "will have",
                       "won't": "will not",
                       "won't've": "will not have",
                       "would've": "would have", "wouldn't": "would not",
                       "wouldn't've": "would not have",
                       "y'all": "you all",
                       "y'all'd": "you all would", "y'all'd've": "you all would have",
                       "y'all're": "you all are",
                       "y'all've": "you all have",
                       "you'd": "you would", "you'd've": "you would have",
                       "you'll": "you will",
                       "you'll've": "you will have",
                       "you're": "you are", "you've": "you have"}




def text_cleaner(text, remove_stopwords = True ):
    '''Remove unwanted characters, stopwords, and format the text to create fewer nulls word embeddings'''
    #convert to text to lower text
    new_string = text.lower()
    new_string = BeautifulSoup(new_string, "lxml").text
    #replace contraction with their long form 
    if True:
        new_string = new_string.split()
        new_text = []
        for word in text:
            if word in CONTRACTION_MAPPING:
                new_text.append(CONTRACTION_MAPPING[word])
            else:
                new_text.append(word)
        new_string = " ".join(new_text)
    
    #Format words and remove unwanted characters    
    new_string= re.sub(r'https?:\/\/.*[\r\n]*', '', new_string, flags=re.MULTILINE)
    new_string = re.sub(r'\<a href', ' ', new_string)
    new_string = re.sub(r'&amp;', '', new_string) 
    new_string = re.sub(r'[_"\-;%()|+&=*%.,!?:#$@\[\]/]', ' ', new_string)
    new_string = re.sub(r'<br />', ' ', new_string)
    new_string = re.sub(r'\'', ' ', new_string)
    
    #remove stopwords
    if remove_stopwords:
        new_string = new_string.split()
        stops = set(stopwords.words("english"))
        new_string = [w for w in new_string if not w in stops]
        new_string = " ".join(new_string)

    return new_string

#clean the summaries and texts

CLEANED_TEXT = []
for t in DATA['content']:
    CLEANED_TEXT.append(text_cleaner(t))

CLEANED_SUMMARY = []
for t in DATA['title']:
    CLEANED_SUMMARY.append(text_cleaner(t, remove_stopwords=False))

DATA['CLEANED_TEXT'] = CLEANED_TEXT
DATA['CLEANED_SUMMARY'] = CLEANED_SUMMARY


def count_words(count_dict, text):
    '''Count the number of occurrences of each word in a set of text'''
    for sentence in text:
        for word in sentence.split():
            if word not in count_dict:
                count_dict[word] = 1
            else:
                count_dict[word] += 1
                
#find the number of times a word appears in the text and the size of the vocabulary

word_counts = {}

count_words(word_counts, CLEANED_SUMMARY)
count_words(word_counts, CLEANED_TEXT)
            
Size_of_Vocabulary= len(word_counts)



# Load Conceptnet Numberbatch's (CN) embeddings, similar to GloVe, but probably better 
# (https://github.com/commonsense/conceptnet-numberbatch)
embeddings_index = {}
with open('https://conceptnet.s3.amazonaws.com/downloads/2019/numberbatch/numberbatch-19.08.txt.gz', encoding='utf-8') as f:
    for line in f:
        values = line.split(' ')
        word = values[0]
        embedding = np.asarray(values[1:], dtype='float32')
        embeddings_index[word] = embedding

Word_embeddings= len(embeddings_index)



# Find the number of words that are missing from CN, and are used more than our threshold.
missing_words = 0
threshold = 20

for word, count in word_counts.items():
    if count > threshold:
        if word not in embeddings_index:
            missing_words += 1
            
missing_ratio = round(missing_words/len(word_counts),4)*100
            
#print("Number of words missing from CN:", missing_words)
#print("Percent of words that are missing from vocabulary: {}%".format(missing_ratio))


# Limit the vocab that we will use to words that appear ≥ threshold or are in GloVe

#dictionary to convert words to integers
vocab_to_int = {} 

value = 0
for word, count in word_counts.items():
    if count >= threshold or word in embeddings_index:
        vocab_to_int[word] = value
        value += 1

# Special tokens that will be added to our vocab
codes = ["<UNK>","<PAD>","<EOS>","<GO>"]   

# Add codes to vocab
for code in codes:
    vocab_to_int[code] = len(vocab_to_int)

# Dictionary to convert integers to words
int_to_vocab = {}
for word, value in vocab_to_int.items():
    int_to_vocab[value] = word

usage_ratio = round(len(vocab_to_int) / len(word_counts),4)*100

len_word_counts = len(word_counts)
len_vocab_to_int = len(vocab_to_int)




# Need to use 300 for embedding dimensions to match CN's vectors.
embedding_dim = 300
nb_words = len(vocab_to_int)

# Create matrix with default values of zero
word_embedding_matrix = np.zeros((nb_words, embedding_dim), dtype=np.float32)
for word, i in vocab_to_int.items():
    if word in embeddings_index:
        word_embedding_matrix[i] = embeddings_index[word]
    else:
        # If word not in CN, create a random embedding for it
        new_embedding = np.array(np.random.uniform(-1.0, 1.0, embedding_dim))
        embeddings_index[word] = new_embedding
        word_embedding_matrix[i] = new_embedding

# Check if value matches len(vocab_to_int)
len_embedding_matrix = len(word_embedding_matrix)
        
def convert_to_ints(text, word_count, unk_count, eos=False):
    '''Convert words in text to an integer.
       If word is not in vocab_to_int, use UNK's integer.
       Total the number of words and UNKs.
       Add EOS token to the end of texts'''
    ints = []
    for sentence in text:
        sentence_ints = []
        for word in sentence.split():
            word_count += 1
            if word in vocab_to_int:
                sentence_ints.append(vocab_to_int[word])
            else:
                sentence_ints.append(vocab_to_int["<UNK>"])
                unk_count += 1
        if eos:
            sentence_ints.append(vocab_to_int["<EOS>"])
        ints.append(sentence_ints)
    return ints, word_count, unk_count

# Apply convert_to_ints to clean_summaries and clean_texts
word_count = 0
unk_count = 0

int_summaries, word_count, unk_count = convert_to_ints(CLEANED_SUMMARY, word_count, unk_count)
int_texts, word_count, unk_count = convert_to_ints(CLEANED_TEXT, word_count, unk_count, eos=True)

unk_percent = round(unk_count/word_count,4)*100

words_in_headlines = word_count
UNKs_in_headlines = unk_count
Percent_words_UNK = unk_percent


def create_lengths(text):
    '''Create a data frame of the sentence lengths from a text'''
    lengths = []
    for sentence in text:
        lengths.append(len(sentence))
    return pd.DataFrame(lengths, columns=['counts'])

lengths_summaries = create_lengths(int_summaries)
lengths_texts = create_lengths(int_texts)

lengths_summaries.describe()
lengths_texts.describe()

def unk_counter(sentence):
    '''Counts the number of time UNK appears in a sentence.'''
    unk_count = 0
    for word in sentence:
        if word == vocab_to_int["<UNK>"]:
            unk_count += 1
    return unk_count

# Sort the summaries and texts by the length of the texts, shortest to longest
# Limit the length of summaries and texts based on the min and max ranges.
# Remove reviews that include too many UNKs

sorted_summaries = []
sorted_texts = []
max_text_length = 84
max_summary_length = 13
min_length = 2
unk_text_limit = 1
unk_summary_limit = 0

for length in range(min(lengths_texts.counts), max_text_length): 
    for count, words in enumerate(int_summaries):
        if (len(int_summaries[count]) >= min_length and
            len(int_summaries[count]) <= max_summary_length and
            len(int_texts[count]) >= min_length and
            unk_counter(int_summaries[count]) <= unk_summary_limit and
            unk_counter(int_texts[count]) <= unk_text_limit and
            length == len(int_texts[count])
           ):
            sorted_summaries.append(int_summaries[count])
            sorted_texts.append(int_texts[count])
        
# Compare lengths to ensure they match
len_sorted_summaries = len(sorted_summaries)
len_sorted_texts = len(sorted_texts)


#building the model
def model_inputs():
    '''Create palceholders for inputs to the model'''
    
    input_data = tf.placeholder(tf.int32, [None, None], name='input')
    targets = tf.placeholder(tf.int32, [None, None], name='targets')
    lr = tf.placeholder(tf.float32, name='learning_rate')
    keep_prob = tf.placeholder(tf.float32, name='keep_prob')
    summary_length = tf.placeholder(tf.int32, (None,), name='summary_length')
    max_summary_length = tf.reduce_max(summary_length, name='max_dec_len')
    text_length = tf.placeholder(tf.int32, (None,), name='text_length')

    return input_data, targets, lr, keep_prob, summary_length, max_summary_length, text_length


def process_encoding_input(target_data, vocab_to_int, batch_size):
    '''Remove the last word id from each batch and concat the <GO> to the begining of each batch'''
    ending = tf.strided_slice(target_data, [0, 0], [batch_size, -1], [1, 1])
    dec_input = tf.concat([tf.fill([batch_size, 1], vocab_to_int['<GO>']), ending], 1)
    return dec_input

def encoding_layer(rnn_size, sequence_length, num_layers, rnn_inputs, keep_prob):
    '''Create the encoding layer'''
    for layer in range(num_layers):
        with tf.variable_scope('encoder_{}'.format(layer)):
            cell_fw = tf.contrib.rnn.LSTMCell(rnn_size,
                                              initializer=tf.random_uniform_initializer(-0.1, 0.1, seed=2))
            cell_fw = tf.contrib.rnn.DropoutWrapper(cell_fw, 
                                                    input_keep_prob = keep_prob)
            cell_bw = tf.contrib.rnn.LSTMCell(rnn_size,
                                              initializer=tf.random_uniform_initializer(-0.1, 0.1, seed=2))
            cell_bw = tf.contrib.rnn.DropoutWrapper(cell_bw, 
                                                    input_keep_prob = keep_prob)
            enc_output, enc_state = tf.nn.bidirectional_dynamic_rnn(cell_fw, 
                                                                    cell_bw, 
                                                                    rnn_inputs,
                                                                    sequence_length,
                                                                    dtype=tf.float32)
             # Join outputs since we are using a bidirectional RNN\n",
    enc_output = tf.concat(enc_output,2)
    
    return enc_output, enc_state
    

def training_decoding_layer(dec_embed_input, summary_length, dec_cell, initial_state, output_layer, 
                            vocab_size, max_summary_length):
    
    '''Create the training logits'''
    
    training_helper = tf.contrib.seq2seq.TrainingHelper(inputs=dec_embed_input,
                                                        sequence_length=summary_length,
                                                        time_major=False)
    
    training_decoder = tf.contrib.seq2seq.BasicDecoder(dec_cell,
                                                       training_helper,
                                                       initial_state,
                                                       output_layer) 
    
    training_logits, _ , _ = tf.contrib.seq2seq.dynamic_decode(training_decoder,
                                                               output_time_major=False,
                                                               impute_finished=True,
                                                               maximum_iterations=max_summary_length)
    return training_decoder


def inference_decoding_layer(embeddings, start_token, end_token, dec_cell, initial_state, output_layer,
                                 max_summary_length, batch_size):
   
    '''Create the inference logits'''
    
    start_tokens = tf.tile(tf.constant([start_token], dtype=tf.int32), [batch_size], name='start_tokens')
    
    inference_helper = tf.contrib.seq2seq.GreedyEmbeddingHelper(embeddings,
                                                                start_tokens,
                                                                end_token)
    inference_decoder = tf.contrib.seq2seq.BasicDecoder(dec_cell,
                                                        inference_helper,
                                                        initial_state,
                                                        output_layer)
    inference_logits, _ , _ = tf.contrib.seq2seq.dynamic_decode(inference_decoder,
                                                                output_time_major=False,
                                                                impute_finished=True,
                                                                maximum_iterations=max_summary_length)
    return inference_decoder
    
    
def decoding_layer(dec_embed_input, embeddings, enc_output, enc_state, vocab_size, text_length, summary_length, 
                   max_summary_length, rnn_size, vocab_to_int, keep_prob, batch_size, num_layers):
    '''Create the decoding cell and attention for the training and inference decoding layers'''
    
    for layer in range(num_layers):
        with tf.variable_scope('decoder_{}'.format(layer)):
            lstm = tf.contrib.rnn.LSTMCell(rnn_size,
                                           initializer=tf.random_uniform_initializer(-0.1, 0.1, seed=2))
            dec_cell = tf.contrib.rnn.DropoutWrapper(lstm, 
                                                     input_keep_prob = keep_prob)
    output_layer = Dense(vocab_size,
                         kernel_initializer = tf.truncated_normal_initializer(mean = 0.0, stddev=0.1))
    attn_mech = tf.contrib.seq2seq.BahdanauAttention(rnn_size,
                                                     enc_output,
                                                     text_length,
                                                     normalize=False,
                                                     name='BahdanauAttention')
    dec_cell = tf.contrib.seq2seq.AttentionWrapper(dec_cell,
                                                   attn_mech,
                                                   rnn_size)
                
    #initial_state = tf.contrib.seq2seq.AttentionWrapperState(enc_state[0],
    #                                                               _zero_state_tensors(rnn_size, 
    #                                                                                   batch_size, 
    #                                                                                   tf.float32)) 
    initial_state = dec_cell.zero_state(batch_size=batch_size,dtype=tf.float32).clone(cell_state=enc_state[0])
    
    with tf.variable_scope("decode"):
        training_decoder = training_decoding_layer(dec_embed_input, 
                                                     summary_length, 
                                                     dec_cell, 
                                                     initial_state,
                                                     output_layer,
                                                     vocab_size, 
                                                     max_summary_length)
        training_logits,_ ,_ = tf.contrib.seq2seq.dynamic_decode(training_decoder,
                                                                  output_time_major=False,
                                                                  impute_finished=True,
                                                                  maximum_iterations=max_summary_length)
         
         
    with tf.variable_scope("decode", reuse=True):
        inference_decoder = inference_decoding_layer(embeddings,  
                                                     vocab_to_int['<GO>'], 
                                                     vocab_to_int['<EOS>'],
                                                     dec_cell, 
                                                     initial_state, 
                                                     output_layer,
                                                     max_summary_length,
                                                     batch_size)
        inference_logits,_ ,_ = tf.contrib.seq2seq.dynamic_decode(inference_decoder,
                                  output_time_major=False,
                                  impute_finished=True,
                                  maximum_iterations=max_summary_length)
    
    return training_logits, inference_logits

def seq2seq_model(input_data, target_data, keep_prob, text_length, summary_length, max_summary_length, 
                  vocab_size, rnn_size, num_layers, vocab_to_int, batch_size):
    
    '''Use the previous functions to create the training and inference logits'''
    # Use Numberbatch's embeddings and the newly created ones as our embeddings
    embeddings = word_embedding_matrix
    
    enc_embed_input = tf.nn.embedding_lookup(embeddings, input_data)
    
    enc_output, enc_state = encoding_layer(rnn_size, text_length, num_layers, enc_embed_input, keep_prob)
    
    dec_input = process_encoding_input(target_data, vocab_to_int, batch_size)
    
    dec_embed_input = tf.nn.embedding_lookup(embeddings, dec_input)
    
    training_logits, inference_logits  = decoding_layer(dec_embed_input, 
                                                        embeddings,
                                                        enc_output,
                                                        enc_state, 
                                                        vocab_size, 
                                                        text_length, 
                                                        summary_length, 
                                                        max_summary_length,
                                                        rnn_size, 
                                                        vocab_to_int, 
                                                        keep_prob, 
                                                        batch_size,
                                                        num_layers)
    return training_logits, inference_logits

def pad_sentence_batch(sentence_batch):
    """Pad sentences with <PAD> so that each sentence of a batch has the same length"""
    max_sentence = max([len(sentence) for sentence in sentence_batch])
    return [sentence + [vocab_to_int['<PAD>']] * (max_sentence - len(sentence)) for sentence in sentence_batch]

def get_batches(summaries, texts, batch_size):
    """Batch summaries, texts, and the lengths of their sentences together"""
    for batch_i in range(0, len(texts)//batch_size):
        start_i = batch_i * batch_size
        summaries_batch = summaries[start_i:start_i + batch_size]
        texts_batch = texts[start_i:start_i + batch_size]
        pad_summaries_batch = np.array(pad_sentence_batch(summaries_batch))
        pad_texts_batch = np.array(pad_sentence_batch(texts_batch))
        # Need the lengths for the _lengths parameters
        pad_summaries_lengths = []
        for summary in pad_summaries_batch:
            pad_summaries_lengths.append(len(summary))
        pad_texts_lengths = []
        for text in pad_texts_batch:
            pad_texts_lengths.append(len(text))
        
        yield pad_summaries_batch, pad_texts_batch, pad_summaries_lengths, pad_texts_lengths

# Set the Hyperparameters
epochs = 100
batch_size = 64
rnn_size = 256
num_layers = 2
learning_rate = 0.005
keep_probability = 0.75

# Build the graph
train_graph = tf.Graph()
# Set the graph to default to ensure that it is ready for training

with train_graph.as_default():
    # Load the model inputs
    input_data, targets, lr, keep_prob, summary_length, max_summary_length, text_length = model_inputs()
    # Create the training and inference logits
    training_logits, inference_logits = seq2seq_model(tf.reverse(input_data, [-1]),
                                                      targets, 
                                                      keep_prob,   
                                                      text_length,
                                                      summary_length,
                                                      max_summary_length,
                                                      len(vocab_to_int)+1,
                                                      rnn_size, 
                                                      num_layers, 
                                                      vocab_to_int,
                                                      batch_size)
    # Create tensors for the training logits and inference logits
    training_logits = tf.identity(training_logits.rnn_output, 'logits')
    inference_logits = tf.identity(inference_logits.sample_id, name='predictions')
    # Create the weights for sequence_loss
    masks = tf.sequence_mask(summary_length, max_summary_length, dtype=tf.float32, name='masks')
    
    with tf.name_scope("optimization"):
        # Loss function
        cost = tf.contrib.seq2seq.sequence_loss(
                training_logits,
                targets,
                masks)
        # Optimizer
        optimizer = tf.train.AdamOptimizer(learning_rate)
        # Gradient Clipping
        gradients = optimizer.compute_gradients(cost)
        capped_gradients = [(tf.clip_by_value(grad, -5., 5.), var) for grad, var in gradients if grad is not None]
        train_op = optimizer.apply_gradients(capped_gradients)
print("Graph is built.")

#training the model
# the training of the model starts from  a subset inorder not to make it easy for the model
#The texts used are closer to the median lengths

# Subset the data for training
start = 200000
end = start + 50000
sorted_summaries_short = sorted_summaries[start:end]
sorted_texts_short = sorted_texts[start:end]
print("The shortest text length:", len(sorted_texts_short[0]))
print("The longest text length:",len(sorted_texts_short[-1]))

# Train the Model
learning_rate_decay = 0.95
min_learning_rate = 0.0005
display_step = 20 # Check training loss after every 20 batches
stop_early = 0 
stop = 3 # If the update loss does not decrease in 3 consecutive update checks, stop training
per_epoch = 3 # Make 3 update checks per epoch
update_check = (len(sorted_texts_short)//batch_size//per_epoch)-1

update_loss = 0 
batch_loss = 0
summary_update_loss = [] # Record the update losses for saving improvements in the model

tf.reset_default_graph()
checkpoint = "best_model.ckpt" 
with tf.Session(graph=train_graph) as sess:
    sess.run(tf.global_variables_initializer())
    # to continue training a previous session
    # loader = tf.train.import_meta_graph(checkpoint + '.meta')
    # loader.restore(sess, checkpoint)
    #sess.run(tf.local_variables_initializer())
    for epoch_i in range(1, epochs+1):
        update_loss = 0
        batch_loss = 0
        for batch_i, (summaries_batch, texts_batch, summaries_lengths, texts_lengths) in enumerate(
                 get_batches(sorted_summaries_short, sorted_texts_short, batch_size)):
            
            start_time = time.time()
            _, loss = sess.run(
                 [train_op, cost],
                 {input_data: texts_batch,
                  targets: summaries_batch,
                  lr: learning_rate,
                  summary_length: summaries_lengths,
                  text_length: texts_lengths,
                  keep_prob: keep_probability})
            batch_loss += loss
            update_loss += loss
            end_time = time.time()
            batch_time = end_time - start_time
            if batch_i % display_step == 0 and batch_i > 0:
                print('Epoch {:>3}/{} Batch {:>4}/{} - Loss: {:>6.3f}, Seconds: {:>4.2f}'
                      .format(epoch_i,
                              epochs, 
                              batch_i, 
                              len(sorted_texts_short) // batch_size, 
                              batch_loss / display_step, 
                              batch_time*display_step))
                batch_loss = 0
                
                #saver = tf.train.Saver() 
                #saver.save(sess, checkpoint)
            if batch_i % update_check == 0 and batch_i > 0:
                print("Average loss for this update:", round(update_loss/update_check,3))
                summary_update_loss.append(update_loss)
                # If the update loss is at a new minimum, save the model
                if update_loss <= min(summary_update_loss):
                    print('New Record!')
                    stop_early = 0
                    saver = tf.train.Saver() 
                    saver.save(sess, checkpoint)
                else:
                    print("No Improvement.")
                    stop_early += 1
                    if stop_early == stop:
                        break
                update_loss = 0
         # Reduce learning rate, but not below its minimum value
        learning_rate *= learning_rate_decay
        if learning_rate < min_learning_rate:
            learning_rate = min_learning_rate      
        if stop_early == stop:
            print("Stopping Training.")
            break

checkpoint = "./best_model.ckpt"
loaded_graph = tf.Graph()
with tf.Session(graph=loaded_graph) as sess:
    # Load saved model
    loader = tf.train.import_meta_graph(checkpoint + '.meta')
    loader.restore(sess, checkpoint)
    names = []
    [names.append(n.name) for n in loaded_graph.as_graph_def().node]
names


#making summaries
#to see the quality of the summaries this model can generate,use a article  
# the length of the summary can be set to a fixed value, or use a random value as below.

def text_to_seq(text):
    '''Prepare the text for the model'''
    text = text_cleaner(text)
    return [vocab_to_int.get(word, vocab_to_int['<UNK>']) for word in text.split()]
                    
                        

    
# use an article
#input_sentence = "I have never eaten an apple before, but this red one was nice. \
                  #I think that I will try a green apple next time."
#text = text_to_seq(input_sentence)
random = np.random.randint(0,len(CLEANED_TEXT))
input_sentence = CLEANED_TEXT[random]
text = text_to_seq(CLEANED_TEXT[random])

checkpoint = "./best_model.ckpt"

loaded_graph = tf.Graph()
with tf.Session(graph=loaded_graph) as sess:
    # Load saved model
    loader = tf.train.import_meta_graph(checkpoint + '.meta')
    loader.restore(sess, checkpoint)

    input_data = loaded_graph.get_tensor_by_name('input:0')
    logits = loaded_graph.get_tensor_by_name('predictions:0')
    text_length = loaded_graph.get_tensor_by_name('text_length:0')
    summary_length = loaded_graph.get_tensor_by_name('summary_length:0')
    keep_prob = loaded_graph.get_tensor_by_name('keep_prob:0')
    
    #Multiply by batch_size to match the model's input parameters
    answer_logits = sess.run(logits, {input_data: [text]*batch_size,
                                      summary_length: [np.random.randint(5,8)],
                                      text_length: [len(text)]*batch_size,
                                      keep_prob: 1.0})[0]

# Remove the padding from the article
pad = vocab_to_int["<PAD>"] 

print('Original Text:', input_sentence)

print('\nText')
print('  Word Ids:    {}'.format([i for i in text]))
print('  Input Words: {}'.format(" ".join([int_to_vocab[i] for i in text])))

print('\nSummary')
print('  Word Ids:       {}'.format([i for i in answer_logits if i != pad]))
print('  Response Words: {}'.format(" ".join([int_to_vocab[i] for i in answer_logits if i != pad])))


    
    
         
    
        
    
    
            


