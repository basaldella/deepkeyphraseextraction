from data.datasets import Hulth

data = Hulth("data/Hulth2003")

train_doc, train_answer = data.load_train()

print("Using %s" % data.__str__())