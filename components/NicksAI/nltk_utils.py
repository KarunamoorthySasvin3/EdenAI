import nltk
import numpy
from nltk.stem.porter import PorterStemmer
stemmer = PorterStemmer()

def tokenize(sentence):
    return nltk.word_tokenize(sentence)

def stem(word):
    return stemmer.stem(word.lower())

def bag_of_words(tokened, words):
    tokenized_sentence = [stem(word) for word in tokened]
    
    bag = numpy.zeros(len(words), dtype=numpy.float32)
    for idx, word in enumerate(words):
        if word in tokenized_sentence:
            bag[idx] = 1.0
            
    return bag