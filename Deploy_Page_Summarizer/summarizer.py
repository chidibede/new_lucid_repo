'''
An Extractive text summarizer that uses
the importance and frequency of a word to summarize

'''
import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from nltk.tokenize import word_tokenize
nltk.download('stopwords')
nltk.download('punkt')

def generate_frequency_table(text):
    ''' This functions creates a frequency
        table of the words in the text and returns
        the frequency table dictionary
    '''
    frequency_table = dict()
    stop_words = set(stopwords.words('english'))
    stemmer = PorterStemmer()
    tokenized_words = word_tokenize(text)

    for word in tokenized_words:
        word = stemmer.stem(word)
        if word in stop_words:
            continue
        if word in frequency_table:
            frequency_table[word] += 1
        else:
            frequency_table[word] = 1
    return frequency_table

def rank_sentences(sentences, frequency_table):
    '''
      This function ranks sentences by virtue of the freuency of words
      appearing in each sentence. Document frequency concept
    '''
    sentence_value = dict()

    for sentence in sentences:
        no_of_words = len(word_tokenize(sentence))
        for word in frequency_table:
            if word in sentence.lower():
                if sentence[:10] in sentence_value:
                    sentence_value[sentence[:10]] += frequency_table[word]
                else:
                    sentence_value[sentence[:10]] = frequency_table[word]
        sentence_value[sentence[:10]] = sentence_value[sentence[:10]] // no_of_words
    return sentence_value

def calc_average_rank(sentence_value):
    '''
        Average value of a sentence from original text
    '''
    sum_values = 0

    for value in sentence_value:
        sum_values += sentence_value[value]

    average = int(sum_values / len(sentence_value))
    return average

def summarize(sentences, sentence_value, threshold):
    '''
     Function that summarizes a text by taking
     the most frequent words
    '''
    sentence_count = 0
    summary = ''

    for sentence in sentences:
        if sentence[:10] in sentence_value and sentence_value[sentence[:10]] > (threshold):
            summary += " " + sentence
            sentence_count += 1

    return summary
