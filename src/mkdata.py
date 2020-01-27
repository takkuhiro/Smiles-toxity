#coding: utf-8
import csv
in_path = "./washed-nr-er.tsv"
out_path0 = "./w0_space.txt"
out_path1 = "./w1_space.txt"
txt0, txt1 = "", ""

f = open(in_path, 'r')
tsv = csv.reader(f, delimiter = '\t')

for i, row in enumerate(tsv):
    if i==0:continue    #1行目は除外
    tmp = ""
    for s in row[0]:
        tmp += s + ' '
    tmp.strip()
    if row[-1] == "0": txt0 += tmp + "\n"
    elif row[-1] == "1": txt1 += tmp + "\n"
    else: print("Error row exists!", i, "行目")

f.close()

with open(out_path0, "w") as f:
    f.write(txt0)

with open(out_path1, "w") as f:
    f.write(txt1)
#
# for sen in txt:
#     tmp = sen.split()
#     print(tmp[0])
