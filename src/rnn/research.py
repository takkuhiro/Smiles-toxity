import os, sys

file0 = '../w0.txt'
file1 = '../w1.txt'

w2ia, w2ib = {}, {}

with open(file0, 'r') as f:
    t0 = f.read().split('\n')

with open(file1, 'r') as f:
    t1 = f.read().split('\n')

idx = 0
for line in t0:
    for s in line:
        if s not in w2ia:
            w2ia[s] = idx
            idx += 1
#print('w0.txt:  ', w2ia)

idx = 0
for line in t1:
    for s in line:
        if s not in w2ib:
            w2ib[s] = idx
            idx += 1
#print('w1.txt:  ', w2ib)

only0, only1 = [], []
for k in w2ia.keys():
    if k not in w2ib:
        only0.append(k)
for k in w2ib.keys():
    if k not in w2ia:
        only1.append(k)

print('w0.txtにだけ出現： ', only0)
print('w1.txtにだけ出現： ', only1)
