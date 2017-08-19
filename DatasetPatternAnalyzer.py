from collections import Counter
from data.datasets import *
from eval import metrics
from nlp import chunker, tokenizer as tk
from utils import info

import nltk

# LOGGING CONFIGURATION

logging.basicConfig(
    format='%(asctime)s\t%(levelname)s\t%(message)s',
    level=logging.DEBUG)

info.log_versions()

# END LOGGING CONFIGURATION


# Dataset and hyperparameters for each dataset

DATASET = Semeval2017

if DATASET == Semeval2017:
    tokenizer = tk.tokenizers.nltk
    DATASET_FOLDER = "data/Semeval2017"
elif DATASET == Hulth:
    tokenizer = tk.tokenizers.nltk
    DATASET_FOLDER = "data/Hulth2003"
else:
    raise NotImplementedError("Can't set the hyperparameters: unknown dataset")


# END PARAMETERS

logging.info("Loading dataset...")

data = DATASET(DATASET_FOLDER)

train_doc_str, train_answer_str = data.load_train()
test_doc_str, test_answer_str = data.load_test()
val_doc_str, val_answer_str = data.load_validation()

train_doc, train_answer = tk.tokenize_set(train_doc_str,train_answer_str,tokenizer)
test_doc, test_answer = tk.tokenize_set(test_doc_str,test_answer_str,tokenizer)
val_doc, val_answer = tk.tokenize_set(val_doc_str,val_answer_str,tokenizer)

logging.info("Dataset loaded. Generating candidate keyphrases...")

train_candidates = chunker.extract_candidates_from_set(train_doc_str,tokenizer)
test_candidates = chunker.extract_candidates_from_set(test_doc_str,tokenizer)
val_candidates = chunker.extract_candidates_from_set(val_doc_str,tokenizer)

logging.debug("Candidates recall on training set   : %.4f", metrics.recall(train_answer,train_candidates))
logging.debug("Candidates recall on test set       : %.4f", metrics.recall(test_answer,test_candidates))
logging.debug("Candidates recall on validation set : %.4f", metrics.recall(val_answer,val_candidates))

train_pos = []
for answers in train_answer.values():
    for answer in answers:
        train_pos.append(nltk.pos_tag(answer))

test_pos = []
for answers in test_answer.values():
    for answer in answers:
        test_pos.append(nltk.pos_tag(answer))

val_pos = []
for answers in val_answer.values():
    for answer in answers:
        val_pos.append(nltk.pos_tag(answer))


train_seq = []
for seq in train_pos:
    pattern = []
    for pos in seq:
        pattern.append(pos[1])
    train_seq.append(' '.join(pattern))

test_seq = []
for seq in test_pos:
    pattern =  []
    for pos in seq:
        pattern.append(pos[1])
    test_seq.append(' '.join(pattern))


val_seq = []
for seq in val_pos:
    pattern = []
    for pos in seq:
        pattern.append(pos[1])
    val_seq.append(' '.join(pattern))

counts = Counter(train_seq + test_seq + val_seq)

for pattern, value in counts.items():
    print("%s \t %s \t occurrences" % (pattern, value))