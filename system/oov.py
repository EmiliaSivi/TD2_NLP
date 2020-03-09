import pickle
import numpy as np
from pcfg import *
import operator

class OOV():
    def __init__ (self, filepath, lexicon):
        
        self.words, self.embeddings = pickle.load(open(filepath, 'rb'), encoding='latin1')
        self.word2id = {word: i for i, word in enumerate(self.words)}

        self.lexicon = lexicon

    def Levenshtein_distance(self, s1, s2): # s1, s2 are words
        n1 = len(s1)
        n2 = len(s2)
        m = np.zeros((n1,n2))
        for i in range(n1):
            m[i, 0] = i
        for j in range(n2):
            m[0, j] = j
        for i in range(1,n1):
            for j in range(1,n2):
                if(s1[i] == s2[j]):
                    m[i, j] = min(m[i-1, j] + 1, m[i, j-1] + 1, m[i-1, j-1])
                else:
                    m[i, j] = min(m[i-1, j] + 1, m[i, j-1] + 1, m[i-1, j-1] + 1)
        return(m[-1, -1])

    def cosine_similarity(self, word1, word2):
        score = 0
        try:
            vec_word_1 = self.embeddings[self.word2id[word1]]
            vec_word_2 = self.embeddings[self.word2id[word2]]
            score = np.dot(vec_word_1, vec_word_2) / (np.linalg.norm(vec_word_1) * np.linalg.norm(vec_word_2))
        except KeyError:
            score = -2
        return score

    def most_similar_words(self, word, k):
        lev_dist = []
        cos_sim = []
        for w in self.words:
            lev_dist.append(self.Levenshtein_distance(word, w)) # with vocabulary or with lexicon
            if(self.cosine_similarity(word, w) == -2):
                cos_sim.append(0)
            else:
                cos_sim.append(self.cosine_similarity(word, w))
        lev_dist = np.array(lev_dist)
        lev_dist = - (2 * (lev_dist - lev_dist.min()) / (lev_dist.max() - lev_dist.min()) - 1)
        cos_sim = np.array(cos_sim)
        s = np.zeros((len(self.words)))
        for i in range(len(self.words)):
            if (cos_sim[i] == 0):
                s[i] = lev_dist[i]
            else:
                s[i] = (lev_dist[i] + cos_sim[i]) / 2
        ind = np.argsort(s)
        # We sort the list of scores in decreasing order
        ind = np.flipud(ind)
        kNN = []
        # We take the k words that maximize the score
        for i in range(1,k+1):
            kNN.append(self.words[ind[i]])
        return(kNN)