from nltk import Tree, Nonterminal, induce_pcfg

class PCFG():
    def __init__ (self, data):

        self.context_free_grammar = self.build_context_free_grammar(data) # a probabilistic context-free grammar whose terminals are part-of-speech tags
        
        self.lexicon = {} # a probabilistic lexicon, i.e. triples of the form (token, part-of-speech tag, probability)
        self.inverse_lexicon = {}
        self.tokens = {}
        self.tags = {}
        self.probs = {}
        self.build_lexicons_and_tokens(data)

    def build_context_free_grammar(self, data):
        productions = []
        for tree in [Tree.fromstring(tree) for tree in data]:
            tree.collapse_unary(collapsePOS=False)
            tree.chomsky_normal_form(horzMarkov=2)
            productions += tree.productions()
        starting_state = Nonterminal('SENT')
        grammar = induce_pcfg(starting_state, productions)
        return grammar

    def build_lexicons_and_tokens(self, data):
        pr = self.context_free_grammar.productions()
        for i in range(len(pr)):
            tag = pr[i].lhs()
            token = pr[i].rhs()
            prb = pr[i].prob()
            if isinstance(tag, Nonterminal) and isinstance(token[0], str):
                try:
                    self.lexicon[token[0]] = {tag: prb, **self.lexicon[token[0]]}
                except KeyError:
                    self.lexicon = {token[0]: {tag: prb}, **self.lexicon}
                try:
                    self.inverse_lexicon[tag] = {token[0]: prb, **self.inverse_lexicon[tag]}
                except KeyError:
                    self.inverse_lexicon[tag] = {token[0]: prb}
            try:
                if isinstance(token[0], Nonterminal) and isinstance(token[1], Nonterminal):
                    try:
                        self.tokens[tag].append(token)
                        self.probs[tag] = {token: prb, **self.probs[tag]}
                    except KeyError:
                        self.tokens[tag] = [token]
                        self.probs[tag] = {token: prb}
                    try:
                        self.tags[token[0]].append(tag)
                    except KeyError:
                        self.tags[token[0]] = [tag]
            except IndexError:
                continue;
        
        # Normalization to obtain sum of probabilities to 1
        
        for token in self.lexicon:
            somme = sum(self.lexicon[token].values())
            for tag in self.lexicon[token]:
                self.lexicon[token][tag] = self.lexicon[token][tag]/somme
        
        # Normalization to obtain sum of probabilities to 1
        
        for tag in self.inverse_lexicon:
            somme = sum(self.inverse_lexicon[tag].values())
            for token in self.inverse_lexicon[tag]:
                self.inverse_lexicon[tag][token] = self.inverse_lexicon[tag][token]/somme
                
        # Normalization to obtain sum of probabilities to 1
                
        for tag in self.probs:
            somme = sum(self.probs[tag].values())
            for token in self.probs[tag]:
                self.probs[tag][token] = self.probs[tag][token]/somme