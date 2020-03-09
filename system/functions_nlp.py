import time
import argparse
import pickle
import re
import numpy as np
import matplotlib.pyplot as plt
from nltk import Tree, Nonterminal, induce_pcfg

def preprocessing_data(data):
    new_data = []
    for sent in data: # [s.strip()[2:-1] for s in data]:
        # Cancel first "( " in the sentence and ")\n" at the end of each sentence
        sent = sent[2:-2]
        # ".": matches any character except a newline
        # "?": after the qualifier makes it perform the match in minimal fashion (as few characters as possible will be matched)
        # Return all non-overlapping matches of pattern in string, as a list of strings
        matches = re.findall("-.+? ", sent)
        # On ne prend pas les mots qui ont un tiret mais qui se termine par une paranthèse car il s'agit de vrais mots à garder complètement
        # Cancel the hyphens
        for m in matches:
            if(m[-2] != ')'):
                # Return the string obtained by replacing the leftmost non-overlapping occurrences of pattern in string by the replacement
                sent = re.sub(m, " ", sent)
        new_data.append(sent)
    return new_data

def get_unparsed(data, ponctuation_fr = False):
    unparsed = []
    for s in data:
        if ponctuation_fr:
            ponctuation = False
        sent = ''
        for token in s.split(' '):
            if '(' in token:
                if ponctuation_fr:
                    if token == '(PONCT':
                        ponctuation = True
                    continue;
            else:
                word = ''
                i = 0
                while(i < len(token) and token[i] != ')'):
                    word += token[i]
                    i+=1
                if ponctuation_fr:
                    if ponctuation == True:
                        sent = sent[:-1]
                        ponctuation = False
                sent = sent + word + ' '
        sent = sent[:-1]
        unparsed.append(sent)
    return(unparsed)
    
def get_parsed(sentence, back, a, b, c):
    result = ""
    if a!= b:
        if (a, b, c) in back:
            a_max, b_max, c_max = back[a, b, c]
            result = result.join([str(c),' (', get_parsed(sentence, back, a, c_max, a_max),') (', get_parsed(sentence, back, c_max+1, b, b_max),')'])
        else:
            result = result.join([str(c),' ', sentence[a]])
    else:
        result = result.join([str(c),' ', sentence[a]])
    return result