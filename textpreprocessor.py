# -*- coding: utf-8 -*-
"""
Created on Mon Sep 14 13:23:15 2020

@author: RETRO-punk
"""
def tpp(x_fulltext):
    import re
    import nltk
    from nltk.corpus import stopwords
    import string
    def remove_punct(x_data):
        string_lwr = x_data.lower()
        translation_table = dict.fromkeys(map(ord, string.punctuation),' ')
        return string_lwr.translate(translation_table)
    x_fulltext_clear_punct = []
    for i in range(len(x_fulltext)):
        test_data = remove_punct(str(x_fulltext[i]))
        x_fulltext_clear_punct.append(test_data)

    # function to remove stopwords
    def remove_stopwords(x_data):
        pattern = re.compile(r'\b(' + r'|'.join(stopwords.words('english')) + r')\b\s*')
        string2 = pattern.sub(' ', x_data)
        return string2
    
    x_fulltext_clear_stopwords = []
    for i in range(len(x_fulltext)):
        test_data = remove_stopwords(x_fulltext_clear_punct[i])
        x_fulltext_clear_stopwords.append(test_data)
        
    # function for tokenizing
    def tokenize_words(x_data):
        words = nltk.word_tokenize(x_data)
        return words
    x_fulltext_tokenized = []
    for i in range(len(x_fulltext)):
        test_data = tokenize_words(x_fulltext_clear_stopwords[i])
        x_fulltext_tokenized.append(test_data)
    return x_fulltext_tokenized
    
    