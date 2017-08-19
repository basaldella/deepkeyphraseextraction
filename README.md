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

## Datasets

The datasets used are:

* [Hulth, 2003](http://www.aclweb.org/anthology/W03-1028): it contains 2000 documents with 19276 different keyphrases, and these keyphrases have 786 different 
part-of-speech patterns.
* [Semeval 2017](http://aclweb.org/anthology/S17-2091):  it contains 500 documents with 9946 different keyphrases, and these keyphrases have 1689 different 
part-of-speech patterns.


## Reproducibility

To ensure reproducibility of the results, you should use the latest development version of Theano.
At the time of writing, results are fully reproducible using Theano version `0.9.0.dev-c697eeab84e5b8a74908da654b66ec9eca4f1291`.

The recommended way of reproducing the results is installing the latest Python version through Anacaonda.

After [installing Anaconda](https://conda.io/miniconda.html), you should create a virtual environment by typing

```
> conda create -n venv anaconda
```

To activate the environment, run the command
```
> source activate venv
```
To install the packages necessary to run the code in this repository, run 
```
> conda install --yes --file requirements.txt
```
After running your experiments, you should exit the virtual environment by running
```
> source deactivate
```