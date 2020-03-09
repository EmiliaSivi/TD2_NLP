from PYEVALB import scorer
from PYEVALB import parser

from functions_nlp import *

class Evalb_nlp():
    def __init__ (self, Cyk, pcfg, oov, data_test, res_test):
        self.Cyk = Cyk
        self.true_parsed = data_test[0:2]
        self.test_parsed = res_test[0:2]
        
    def get_results(self):
        results = []
        for i in range(len(self.true_parsed)):
            sentence_true = self.true_parsed[i]
            sentence_test = self.test_parsed[i]
            back, a, b, c = self.Cyk.cyk(sentence_test)
            sentence = sentence_test.split(' ')
            result_test = "".join(['((SENT (', get_parsed(sentence, back, a, b, c), ')))'])
            result_test = result_test[1:-1]
            print("Result sentence:")
            print(result_test)
            
            target = parser.create_from_bracket_string(sentence_true)
            predicted = parser.create_from_bracket_string(result_test)
            
            s = scorer.Scorer()
            result = s.score_trees(target, predicted)
            
            print('The recall is: ' + str(result.recall))
            print('The precision is: ' + str(result.prec))
            
            results.append(result_test)
        
        return(results)