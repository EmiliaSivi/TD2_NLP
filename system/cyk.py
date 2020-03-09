# -*- coding: utf-8 -*-
"""
Created on Mon Mar  9 20:40:26 2020

@author: Emilia
"""

import numpy as np
from nltk import Tree, Nonterminal, induce_pcfg
from oov import *
from functions_nlp import *

class CYK():
    def __init__ (self, pcfg, oov):
        
        self.oov = oov
        self.pcfg = pcfg
        self.context_free_grammar = self.pcfg.context_free_grammar
        self.lexicon = self.pcfg.lexicon
        self.inverse_lexicon = self.pcfg.inverse_lexicon
        self.tokens = self.pcfg.tokens
        self.tags = self.pcfg.tags
        self.probs = self.pcfg.probs
        
    def most_similar_tags(self, word):
        L_sim_words = self.oov.most_similar_words(word, 20)
        nonterminal = []
        nonterminal_prb = []
        for i in range(20):
            sim_word = L_sim_words[i]
            if (sim_word in self.lexicon):
                if(len(self.lexicon[sim_word].items())==1):
                    el = [e[0] for e in self.lexicon[sim_word].items()]
                    prb = [e[1] for e in self.lexicon[sim_word].items()]
                    nonterminal.append(el[0])
                    nonterminal_prb.append(prb[0])
                else:
                    max_nt = None
                    max_prb = 0
                    for e in self.lexicon[sim_word].items():
                        prb = e[1]
                        if prb > max_prb:
                            max_prb = prb
                            max_nt = e[0]
                    nonterminal.append(max_nt)
                    nonterminal_prb.append(max_prb)
        return(nonterminal, nonterminal_prb)
    
    def get_bestnonterminal(self, word, k):
        L_nonterminals, L_probs = self.most_similar_tags(word)
        if k == 1:
            if len(L_nonterminals) == 0:
                nonterminal = None
                L_prb = [0]
            else:
                nonterminal = L_nonterminals[0]
                L_prb = L_probs[0]
        else:
            nonterminal = L_nonterminals
            L_prb = L_probs
        return(nonterminal, L_prb)

    def cyk(self, sentence):
        sentence = sentence.split(' ')
        n = len(sentence)
        pr = self.context_free_grammar.productions()
        R = [pr[i].lhs() for i in range(len(pr))] # grammar containing r nonterminal symbols
        start_symbol = R[0]
        r = len(R)
        
        P = {} # P[n,n,r] be an array of real numbers
        back = {} # back[n,n,r] be an array of backpointing triples
        
        for s in range(n):
            a_s = sentence[s]
            if a_s in self.lexicon:
                for tag_s in self.inverse_lexicon:        
                    try:
                        P[s, s, tag_s] = self.lexicon[a_s][tag_s]
                    except:
                        P[s, s, tag_s] = 0
            else:
                a_s, prb_a_s = self.get_bestnonterminal(a_s, 1)
                for tag_s in self.inverse_lexicon:        
                    if tag_s == a_s: 
                        P[s, s, tag_s] = prb_a_s
                    else:
                        P[s, s, tag_s] = 0
        
        for l in range(n-1): # length of span
            for s in range(n-l-1): # start of span
                for a in self.tokens:
                    max_score = 0
                    max_rule = None
                    for i in range(s, s+l+1): # partition of span
                        for b, c in self.tokens[a]: # for each production
                            score = self.probs[a][b, c]
                            try:
                                score = score * P[s, i, b]
                            except:
                                P[s, i, b] = 0
                                score = 0
                                try:
                                    score = score * P[i+1, s+l+1, c]
                                except:
                                    P[i+1, s+l+1, c] = 0
                                    score = 0
                            try:
                                score = score * P[i+1, s+l+1, c]
                            except:
                                P[i+1, s+l+1, c] = 0
                                score = 0
                            if score > max_score:
                                max_score = score
                                max_rule = (b, c, i)
                    P[s, s+l+1, a] = max_score
                    back[s, s+l+1, a] = max_rule
        max_score = 0
        max_rule = None
        for (i, j, a) in P:
            if i == 0 and j == n-1:
                prb = P[i, j, a]
                if prb > max_score:
                    max_score = prb
                    max_rule = (i, j, a)
        
        return(back, *max_rule)