The goal of this project is to build a French probabilistic parser.
The parser is based on the PCFG model and on a probabilistic version of the CYK algorithm.
The probabilistic parser will use an Out Of Vocabulary model robust to unknown words (out of vocabulary words).

The dataset is in the sequoia-corpusfct.txt file.

The output results are in the evaluation_data.parser_output.txt file.

The program can be run by using the run.sh file or by running the main file.
Due to a lack of time, I couldn't add any option to the program:
it could be interesting to be able to give it's own sentence to the program
or to add some other options on how to compute the similarity between two words.