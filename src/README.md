# Deep Keyphrase Extraction

Deep Neural Networks for Keyphrase Extraction.

Currently containing three scripts:
* `SimpleRNN.py` : Bidirectional LSTM recurrent neural network that scans the text and selects the keyphrases.
* `MergeRNN.py`: Bidirectional LSTM recurrent neural network that scans the text two times: the left branch of the network 
reads the text and produces an encoded version of the document. This representation is then merged with the word embedding
 of each word and the text is scanned again using another Bidirectional LSTM that selects the keyphrases.
* `AnswerRNN.py`: inspired from Question Answering models, this network receives a series of candidate keyphrases
generated using part-of-speech tag patterns and compares them with the document. It used two Bidirectional LSTMs to generate the representations
of both the document and the keyphrase and another network on top with classifies each candidate KP.
* `AnswerRNN2.py`: evolution of `AnswerRNN`, borrows from [Feng et Al., 2015](https://arxiv.org/pdf/1508.01585v2.pdf) 
and [Tan et al, 2016](https://arxiv.org/pdf/1511.04108.pdf) similarity-based models. 

## Datasets

The datasets used are:

* [Hulth, 2003](http://www.aclweb.org/anthology/W03-1028): it contains 2000 documents with 19276 different keyphrases, and these keyphrases have 786 different 
part-of-speech patterns.
* [Semeval 2017](http://aclweb.org/anthology/S17-2091):  it contains 500 documents with 9946 different keyphrases, and these keyphrases have 1689 different 
part-of-speech patterns.
* [Kp20k](https://arxiv.org/pdf/1704.06879.pdf):  it contains 20000 documents.




```
