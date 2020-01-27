#coding: utf-8
#最大長を出力
in_path = "./washed-nr-er.tsv"

with open(in_path, 'r') as f:
    txt = f.read().split()
dic = []
ans = 0
for s in txt:
    if ans < len(s):
        ans = len(s)
    if s not in dic:
        dic.append(s)
print(dic)
print(len(dic))
print('最大長: ', ans)
##データ連結
#in_0 = './w0.txt'
#in_1 = './out/newout0.txt'
#out = './out/out0-augment.txt'
#with open(in_0, 'r') as f:
#    tmp = f.read().split()
#
#with open(in_1, 'r') as f:
#    tmp1 = f.read().split()
#
#with open(out, 'w') as f:
#    for line in tmp:
#        f.write(line+'\n')
#    for line in tmp1:
#        f.write(line+'\n')

#tab月のデータフォーマットに揃える．
#in_0 = './out/newout0.txt'
#out_0 = './out/newout0-tab.tsv'
#in_1 = './out/newout1.txt'
#out_1 = './out/newout1-tab.tsv'
#in_0_orig = './w0.txt'
#out_0_orig = './out/w0-tab.tsv'
#in_1_orig = './w1.txt'
#out_1_orig = './out/w1-tab.tsv'
#
#def write(in_f, out_f, num):
#    with open(in_f, 'r') as f:
#        tmp = f.read().split()
#    with open(out_f, 'w') as f:
#        f.write('washed_mol'+'\t'+'Formula'+'\t'+'FW'+'\t'+'DSSTox_CID'+'\t'+'Active'+'\n')
#        for line in tmp:
#            f.write(line + '\t' + 'C' + '\t' + '0'+ '\t' + '0' + '\t' + str(num) + '\n')
#write(in_0, out_0, 0)
#write(in_1, out_1, 1)
#write(in_0_orig, out_0_orig, 0)
#write(in_1_orig, out_1_orig, 1)


#tsvファイルの行数確認
# inp = './washed-nr-er.tsv'
# with open(inp, 'r') as f:
#     tmp = f.read().split('\n')
# for line in tmp:
#     seq = line.split('\t')
#     print(len(seq))
