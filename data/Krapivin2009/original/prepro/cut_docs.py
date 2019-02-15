import os
import re
import json
import collections
import numpy as np

def striphtml(data):
    p = re.compile(r'<.*?>')
    return p.sub('', data)

path = "datasets/Krapivin2009/all_docs_abstacts_refined" 
for f in os.listdir(path):
    if not f.endswith(".txt"):
       continue
    f_text = open(os.path.join(path, f), "rU")
    f_abstr = ''
    for l in f_text:
	if (l == "--B\n"):
	    break
	else:
	    if ((l != "--T\n") and (l != "--A\n")):
	       f_abstr = ''.join([f_abstr, striphtml(l)])

    with open('datasets/Krapivin2009/abstr/'+f,'w') as f: 
         f.write(f_abstr.encode('utf-8'))



