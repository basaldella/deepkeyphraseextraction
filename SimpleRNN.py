from data.datasets import Hulth
import data.datasets as ds
import logging

logging.basicConfig(
    format='%(asctime)s\t%(levelname)s\t%(message)s',
    level=logging.INFO)

data = Hulth("data/Hulth2003")

train_doc, train_answer = data.load_train()
test_doc, test_answer = data.load_test()
train_answer_seq = ds.make_sequential(train_doc,train_answer)

