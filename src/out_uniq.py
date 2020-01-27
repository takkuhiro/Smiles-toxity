#出力テキストと訓練データを比較し，訓練データ中にないものを抽出
import pickle

outfile = './out/out0-1.txt'
originalfile = './w0.txt'
generatefile = './out/out0-1_only_new.txt'

with open(outfile, 'r') as f:
    lines0 = f.read().strip().split('\n')

with open(originalfile, 'r') as f:
    lines1 = f.read().strip().split('\n')

tmp = []
for line in lines0:
    if line not in lines1:
        tmp.append(line)

with open(generatefile, 'w') as f:
    for line in tmp:
        f.write(line+'\n')
        f.flush()
