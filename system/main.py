import time
import argparse
import pickle
import re
import numpy as np
import matplotlib.pyplot as plt

from PYEVALB import scorer # as PYEVALB_scorer
from PYEVALB import parser # as PYEVALB_parse
from nltk import Tree, Nonterminal, induce_pcfg

from functions_nlp import *
from pcfg import *
from oov import *
from cyk import *
from evalb_nlp import *

file_sequoia = open("sequoia-corpusfct.txt", "r", encoding='utf-8')

data = list(file_sequoia)

N = len(data)
n = round(N*0.8)
m = round(N*0.1)

data_train = data[:n]
data_evaluation = data[n+1:n+m]
data_test = data[n+m+1:]

res_train = get_unparsed(data_train)
res_evaluation = get_unparsed(data_evaluation)
res_test = get_unparsed(data_test)

data_train = preprocessing_data(data_train)
data_evaluation = preprocessing_data(data_evaluation)
data_test = preprocessing_data(data_test)

# PCFG and OOV

filepath = 'polyglot-fr.pkl'

pcfg = PCFG(data_train)
oov = OOV(filepath, pcfg.lexicon)
print("K Most Similar Words of: accress")
print(oov.most_similar_words("accress", 5))
print("####################################")
print("K Most Similar Words of: universite")
print(oov.most_similar_words("universite", 5))
print("####################################")
print("\n")

# CYK

Cyk = CYK(pcfg, oov)
print("K most similar tags of: universite")
tags, probs = Cyk.get_bestnonterminal('universite', 5)
print(tags)

evalb_nlp = Evalb_nlp(Cyk, pcfg, oov, data_test, res_test)

results = evalb_nlp.get_results()

file_output_final = open("evaluation_data.parser_output.txt", "w", encoding='utf-8')
file_output_final.writelines(sentence + '\n' for sentence in results)

file_sequoia.close()
file_output_final.close()